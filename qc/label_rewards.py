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

# Ported verbatim from openpi's scripts/add_subgoal_reward_to_qc_cache.py --
# same production thresholds (slurm/train_libero_qc_critic_full_finetune_subgoal.slurm),
# same gripper-qpos state indices (confirmed identical layout here: state[6:8]).
GRIPPER_DIM_1 = 6
GRIPPER_DIM_2 = 7
OPEN_THRESHOLD = 0.06
CLOSED_THRESHOLD = 0.02
MIN_CLOSED_FRAMES = 10
REWARD_VALUE = 1.0


def _gripper_width(state: np.ndarray) -> np.ndarray:
    return state[:, GRIPPER_DIM_1] - state[:, GRIPPER_DIM_2]


def _detect_release_transitions(width: np.ndarray) -> list[int]:
    """Hysteresis detector: open -> (closed for >= MIN_CLOSED_FRAMES) -> open
    again marks a release event at the re-open frame. Identical logic to
    openpi's version -- see that file for the full rationale (confirmed via
    raw trace inspection to correctly avoid threshold-boundary noise)."""
    events = []
    state = "open"
    closed_run = 0
    for i, w in enumerate(width):
        if state == "open":
            if w < CLOSED_THRESHOLD:
                state = "closed"
                closed_run = 1
        elif state == "closed":
            if w < CLOSED_THRESHOLD:
                closed_run += 1
            elif w > OPEN_THRESHOLD:
                if closed_run >= MIN_CLOSED_FRAMES:
                    events.append(i)
                state = "open"
                closed_run = 0
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
    args = p.parse_args()

    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(f"cuda:{args.cuda}").eval()

    ds = ParquetLiberoDataset(args.dataset_root)
    num_episodes = ds.num_episodes if args.max_episodes is None else min(args.max_episodes, ds.num_episodes)
    print(f"Labeling {num_episodes}/{ds.num_episodes} episodes, horizon_length={args.horizon_length}, "
          f"num_candidates={args.num_candidates}")

    out = {}
    skipped = 0
    for ep_idx in tqdm.tqdm(range(num_episodes)):
        result = label_episode(model, ds, ep_idx, args.horizon_length, args.num_candidates)
        if result is None:
            skipped += 1
            continue
        for k, v in result.items():
            out[f"{k}_{ep_idx}"] = v

    out["_num_episodes"] = num_episodes - skipped
    out["_horizon_length"] = args.horizon_length
    out["_num_candidates"] = args.num_candidates
    out["_ckpt_path"] = args.ckpt_path
    np.savez(args.output_path, **out)
    print(f"Wrote {num_episodes - skipped} labeled episodes ({skipped} skipped, too short) to {args.output_path}")


if __name__ == "__main__":
    main()
