"""One-time reward + embedding + candidate-action labeling pass over the
physical-intelligence/libero LeRobot dataset, for training a QChunkCritic
on top of the frozen VLA-JEPA-LIBERO checkpoint.

Mirrors openpi's scripts/qc_label_rewards.py in spirit (same cached-fields
contract: per-episode reward/embed/candidates arrays, chunk-level TD
aggregation deferred to load time so discount/horizon can change without
re-labeling) but differs in the reward signal by deliberate scope decision:
this uses ONLY the gripper-release/subgoal-completion heuristic (ported
verbatim from scripts/add_subgoal_reward_to_qc_cache.py's thresholds), NOT
openpi's JEPA-prediction-error intrinsic reward. That's the eventual goal
(the fundamental idea IS the intrinsic-reward version -- see PR discussion)
but requires porting VLA-JEPA's video encoder + world-model-predictor loss
into this labeling pass too, which is separate, larger follow-up work.
Reward computation lives in its own function (`_subgoal_rewards`) precisely
so it's swappable later without touching the embedding/candidate machinery.

For every frame i in every episode, runs ONE predict_action_candidates call
(qc/sampling.py) to get:
  - embed[i]: pooled embodied_action_tokens (mean over token dim)
  - candidates[i]: num_candidates action chunks sampled from the frozen actor

Transitions are then built by pairing frame t's (embed, action_chunk, reward)
with frame (t+horizon_length)'s (embed, candidates) as the TD target's next
state -- exactly once per frame, not once per transition it participates in,
since a frame's embed/candidates don't depend on which role it plays.

Usage:
    PYTHONPATH=. .venv/bin/python qc/label_rewards.py \
        --dataset-root /path/to/physical-intelligence-libero \
        --ckpt-path /path/to/VLA-JEPA-LIBERO.pt \
        --output-path qc_cache.npz \
        [--horizon-length 5] [--num-candidates 8] [--max-episodes N]
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
from PIL import Image

from qc.sampling import predict_action_candidates
from starVLA.model.framework.base_framework import baseframework

# Ported verbatim from openpi's scripts/add_subgoal_reward_to_qc_cache.py
# initially (same gripper-qpos state indices, state[6:8]), but the fixed
# absolute OPEN_THRESHOLD/CLOSED_THRESHOLD hysteresis detector was found to
# badly under-detect real grasp/release events on this dataset: two
# independent problems, confirmed by direct trace inspection --
#   1. The "closed" width for a real grasp depends on the grasped object's
#      size (a book/soup-can stops the fingers at width~0.03-0.06; a thin
#      object at ~0.01-0.02) -- CLOSED_THRESHOLD=0.02 only ever fires for
#      the thinnest objects. Sample: "pick"-verb episodes showed 100% zero
#      detected release events under the absolute-threshold version.
#   2. Most demonstrations end the recording before the gripper fully
#      reopens (episode terminates right at/soon after task success, not
#      after a deliberate "return to open" motion) -- so even when closing
#      IS detected, the exit condition (w > OPEN_THRESHOLD) often never
#      fires either.
# Both matter most for MULTI-STEP tasks (e.g. "put both X and Y in the
# basket"), which need a release event mid-episode (not just the final-
# frame terminal bonus already added unconditionally in _subgoal_rewards)
# to get any reward signal for completing the FIRST sub-goal -- exactly
# libero_10's long-horizon task style. Fixed via a relative-threshold
# detector: "closed" is a large drop below a slow-decaying rolling-max
# "open" baseline (adapts to whatever width this particular object leaves),
# and a stalled-but-still-closing episode gets credited a release-in-
# progress at its last frame rather than requiring a full return to the
# original open baseline. Validated against 300 real episodes: "pick"
# 100%->0% zero-detection rate, "put" 47%->9%, "turn" 65%->0%, with a sane
# 1-3 events/episode distribution (median 1) and no runaway false positives.
GRIPPER_DIM_1 = 6
GRIPPER_DIM_2 = 7
MIN_CLOSED_FRAMES = 10
DROP_FRAC = 0.3        # "closed" = dropped >= 30% below the recent open baseline
RECOVER_MARGIN = 0.15  # "still closing" = within 15% of the closed plateau's own minimum
REWARD_VALUE = 1.0


def _gripper_width(state: np.ndarray) -> np.ndarray:
    return state[:, GRIPPER_DIM_1] - state[:, GRIPPER_DIM_2]


def _detect_release_transitions(width: np.ndarray) -> list[int]:
    """Relative-threshold hysteresis detector -- see the module-level
    comment above GRIPPER_DIM_1 for the full rationale. NOT dataset-
    portable as-is: this specific relative-margin logic broke on Fractal's
    hard-0/1-clipped gripper signal (see qc/label_rewards_simplerenv.py,
    which uses a different detector for that dataset specifically)."""
    events = []
    state = "open"
    open_baseline = float(width[0])
    closed_run = 0
    closed_min = None
    for i, w in enumerate(width):
        w = float(w)
        if state == "open":
            open_baseline = max(open_baseline * 0.98, w)
            if w < open_baseline * (1 - DROP_FRAC):
                state = "closed"
                closed_run = 1
                closed_min = w
        elif state == "closed":
            closed_min = min(closed_min, w)
            if w < closed_min * (1 + RECOVER_MARGIN) and w <= open_baseline * (1 - DROP_FRAC * 0.5):
                closed_run += 1
            else:
                if closed_run >= MIN_CLOSED_FRAMES:
                    events.append(i)
                state = "open"
                open_baseline = w
                closed_run = 0
    if state == "closed" and closed_run >= MIN_CLOSED_FRAMES:
        # episode ended still closed after a sustained grasp -- credit a
        # release-in-progress rather than losing this transition entirely
        events.append(len(width) - 1)
    return events


def _subgoal_rewards(state: np.ndarray) -> np.ndarray:
    """[T] reward array: REWARD_VALUE at each detected release event plus an
    unconditional bonus on the final frame (task-completion proxy), 0
    elsewhere. Swap this function out to change the reward signal without
    touching anything below."""
    n = state.shape[0]
    reward = np.zeros(n, dtype=np.float32)
    width = _gripper_width(state)
    for e in _detect_release_transitions(width):
        reward[e] = REWARD_VALUE
    reward[n - 1] = REWARD_VALUE
    return reward


def _decode_image(cell: dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


class ParquetLiberoDataset:
    """Reads physical-intelligence/libero's LeRobot v2.0 parquet files
    directly, bypassing the `lerobot` package entirely -- the installed
    lerobot==0.4.4 rejects this dataset's v2.0 codebase_version outright
    (BackwardCompatibilityError), and since this dataset stores per-frame
    images as embedded PNG bytes (not video, total_videos=0), there's no
    video-decoding machinery worth pulling the dependency in for anyway."""

    def __init__(self, root: str):
        self.root = Path(root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.chunks_size = info["chunks_size"]
        self.data_path_template = info["data_path"]
        self.episodes = [json.loads(line) for line in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()]
        self.num_episodes = len(self.episodes)

    def load_episode(self, ep_idx: int) -> tuple[np.ndarray, np.ndarray, list, list, str]:
        chunk = ep_idx // self.chunks_size
        path = self.root / self.data_path_template.format(episode_chunk=chunk, episode_index=ep_idx)
        table = pq.read_table(path)
        rows = table.to_pylist()
        states = np.array([r["state"] for r in rows], dtype=np.float32)
        actions = np.array([r["actions"] for r in rows], dtype=np.float32)
        images = [r["image"] for r in rows]
        wrist_images = [r["wrist_image"] for r in rows]
        task = self.episodes[ep_idx]["tasks"][0]
        return states, actions, images, wrist_images, task


def label_episode(
    model,
    ds: ParquetLiberoDataset,
    ep_idx: int,
    horizon_length: int,
    num_candidates: int,
) -> dict | None:
    states, actions, images, wrist_images, task = ds.load_episode(ep_idx)
    n = states.shape[0]
    if n <= horizon_length:
        return None  # too short for even one full transition

    reward = _subgoal_rewards(states)

    embed_dim = None
    embeds = None
    candidates = None
    for i in range(n):
        img = _decode_image(images[i])
        wrist = _decode_image(wrist_images[i])
        state_i = states[i : i + 1]  # [1, state_dim], matches predict_action's expected shape

        out = predict_action_candidates(
            model, [[img, wrist]], [task], state=[state_i], num_samples=num_candidates
        )
        embed_i = out["embodied_action_tokens"].mean(axis=1)[0]  # [H] pooled
        cand_i = out["normalized_actions"]  # [num_candidates, chunk_len, action_dim]

        if embeds is None:
            embed_dim = embed_i.shape[0]
            embeds = np.zeros((n, embed_dim), dtype=np.float32)
            candidates = np.zeros((n, *cand_i.shape), dtype=np.float32)
        embeds[i] = embed_i
        candidates[i] = cand_i

    return {
        "state": states,
        "action": actions,
        "reward": reward,
        "embed": embeds,
        "candidates": candidates,
        "horizon_length": horizon_length,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--horizon-length", type=int, default=5)
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Write partial progress to --output-path every N episodes, so a crash "
             "loses at most N episodes of work instead of the whole multi-hour run "
             "(this script previously had no checkpointing at all). Overwrites "
             "--output-path in place each time (atomic tmp-then-rename).",
    )
    args = p.parse_args()

    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(f"cuda:{args.cuda}").eval()

    ds = ParquetLiberoDataset(args.dataset_root)
    num_episodes = ds.num_episodes if args.max_episodes is None else min(args.max_episodes, ds.num_episodes)

    # Resume support: if --output-path already exists (from a previous,
    # interrupted run of THIS script), pick up where it left off instead of
    # starting over. _done_episodes tracks which episode indices are
    # finished, mirroring qc/label_rewards_simplerenv.py's _done_keys.
    out_path = Path(args.output_path)
    if out_path.exists():
        out = dict(np.load(out_path, allow_pickle=True))
        done_episodes = set(out.get("_done_episodes", np.array([], dtype=np.int64)).tolist())
        print(f"Resuming from {out_path}: {len(done_episodes)} episodes already done.")
    else:
        out = {}
        done_episodes = set()

    print(f"Labeling {num_episodes}/{ds.num_episodes} episodes, horizon_length={args.horizon_length}, "
          f"num_candidates={args.num_candidates}")

    def save_checkpoint(skipped):
        out["_num_episodes"] = num_episodes - skipped
        out["_horizon_length"] = args.horizon_length
        out["_num_candidates"] = args.num_candidates
        out["_ckpt_path"] = args.ckpt_path
        out["_done_episodes"] = np.array(sorted(done_episodes), dtype=np.int64)
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **out)
        tmp_path.replace(out_path)  # atomic on the same filesystem -- never leaves a half-written output file

    skipped = 0
    for ep_idx in tqdm.tqdm(range(num_episodes)):
        if ep_idx in done_episodes:
            continue
        result = label_episode(model, ds, ep_idx, args.horizon_length, args.num_candidates)
        if result is None:
            skipped += 1
            continue
        for k, v in result.items():
            out[f"{k}_{ep_idx}"] = v
        done_episodes.add(ep_idx)

        if len(done_episodes) % args.checkpoint_every == 0:
            save_checkpoint(skipped)

    save_checkpoint(skipped)
    print(f"Wrote {num_episodes - skipped} labeled episodes ({skipped} skipped, too short) to {args.output_path}")


if __name__ == "__main__":
    main()
