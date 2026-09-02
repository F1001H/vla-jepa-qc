"""Reward + embedding + candidate-action labeling pass for SimplerEnv QC
critic training, over the SAME two OXE datasets VLA-JEPA's paper used for
SimplerEnv post-training (bridge_rt_1 mixture: bridge_orig + fractal20220817,
see starVLA/dataloader/gr00t_lerobot/mixtures.py) -- we don't redo their base
fine-tuning (that's what the released VLA-JEPA-SimplerEnv.pt checkpoint
already is), just label a QC training cache against the frozen checkpoint,
mirroring qc/label_rewards.py's LIBERO pipeline.

Differences from the LIBERO version:
  - SimplerEnv inference uses a SINGLE camera image per step (no wrist
    camera), confirmed from model2simpler_interface.py's step() --
    predict_action_candidates already handles arbitrary image-list shapes,
    no changes needed there.
  - Video-encoded (mp4 via PyAV), not embedded-PNG parquet cells.
  - Gripper reward signal is a single scalar state[:, 7] in [0, 1] (1=open,
    0=closed) for BOTH datasets -- confirmed by inspecting real episode
    traces -- not a two-finger width difference like LIBERO's state[6]-state[7].
    Thresholds/min_closed_frames recalibrated for this scale and for these
    datasets' much shorter episodes (25-115 frames here vs LIBERO's 75-505).
  - snapshot_download() is broken for these two repos (>90k files triggers a
    huggingface_hub library bug: "ValueError: min() arg is an empty
    sequence" during the list_repo_tree fallback, regardless of --include
    filters). Downloads individual files directly via hf_hub_download
    instead, on demand per episode.

Usage:
    PYTHONPATH=. .venv/bin/python qc/label_rewards_simplerenv.py \
        --ckpt-path /path/to/VLA-JEPA-SimplerEnv.pt \
        --output-path qc_cache_simplerenv.npz \
        [--bridge-episodes 500] [--fractal-episodes 500] [--num-candidates 8]
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from qc.sampling import predict_action_candidates
from starVLA.model.framework.base_framework import baseframework

# Calibrated from real traces (see module docstring) -- gripper state in
# [0,1], 1=open, 0=closed. Thresholds set well inside the observed clusters
# (closed traces sit near 0-0.35, open traces near 0.9-1.0).
OPEN_THRESHOLD = 0.8
CLOSED_THRESHOLD = 0.3
MIN_CLOSED_FRAMES = 3  # much lower than LIBERO's 10 -- these episodes are 25-115 frames, not 75-505
REWARD_VALUE = 1.0

# Bridge and Fractal need DIFFERENT release-detection strategies, confirmed
# by direct trace inspection (see qc/label_rewards.py's much longer comment
# for the LIBERO version of this same investigation -- same root problem,
# same fix, ported here):
#   - Bridge: like LIBERO, real "closed" values are object-width-dependent
#     (e.g. grasping broccoli/chocolate only closes the gripper to
#     ~0.55-0.76, never crossing CLOSED_THRESHOLD=0.3) -- absolute
#     thresholds badly under-detect (78% of "pick" episodes showed ZERO
#     release events). Needs the same relative-threshold detector as
#     LIBERO: "closed" = dropped >= DROP_FRAC below a rolling open
#     baseline, credits a release-in-progress at episode end.
#   - Fractal: the OPPOSITE problem -- its signal is hard-clipped to
#     exactly 0.0/1.0 (some episodes even START at 0.0, gripper already
#     closed), which breaks the relative detector's recover-margin check
#     (anything times (1+margin) is still 0 once closed_min hits exactly
#     0, so it can never re-detect "still closed" and fires spurious
#     mid-grasp releases). Fractal's absolute thresholds were ALREADY
#     well-calibrated (0-6% zero-detection rate for pick/place/move) --
#     only needed the same "credit an episode-end release-in-progress"
#     addition, no threshold changes.
# Validated against 400-500 real episodes per dataset before landing this.
DROP_FRAC = 0.3
RECOVER_MARGIN = 0.15

DATASETS = {
    "bridge": {
        "repo_id": "IPEC-COMMUNITY/bridge_orig_lerobot",
        "video_key": "observation.images.image_0",
        "local_dir_name": "bridge_orig_lerobot",
    },
    "fractal": {
        "repo_id": "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "video_key": "observation.images.image",
        "local_dir_name": "fractal20220817_data_lerobot",
    },
}
LOCAL_ROOT = Path.home() / "starvla_jepa" / "datasets"


def _gripper_signal(state: np.ndarray) -> np.ndarray:
    return state[:, 7]


def _detect_release_transitions_absolute(signal: np.ndarray) -> list[int]:
    """Fractal: absolute-threshold hysteresis, already well-calibrated (see
    module comment above DROP_FRAC), plus crediting a release-in-progress
    if the episode ends still closed after a sustained grasp."""
    events = []
    state = "open"
    closed_run = 0
    for i, w in enumerate(signal):
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
    if state == "closed" and closed_run >= MIN_CLOSED_FRAMES:
        events.append(len(signal) - 1)
    return events


def _detect_release_transitions_relative(signal: np.ndarray) -> list[int]:
    """Bridge: relative-threshold hysteresis -- see module comment above
    DROP_FRAC. Identical algorithm to qc/label_rewards.py's LIBERO
    detector; NOT used for Fractal (breaks on its hard-0/1-clipped
    signal, see module comment)."""
    events = []
    state = "open"
    open_baseline = float(signal[0])
    closed_run = 0
    closed_min = None
    for i, w in enumerate(signal):
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
        events.append(len(signal) - 1)
    return events


def _subgoal_rewards(state: np.ndarray, dataset_key: str) -> np.ndarray:
    n = state.shape[0]
    reward = np.zeros(n, dtype=np.float32)
    signal = _gripper_signal(state)
    detector = _detect_release_transitions_relative if dataset_key == "bridge" else _detect_release_transitions_absolute
    for e in detector(signal):
        reward[e] = REWARD_VALUE
    reward[n - 1] = REWARD_VALUE
    return reward


class SimplerEnvEpisodeSource:
    """Downloads (on demand, direct per-file -- see module docstring for
    why not snapshot_download) and decodes one dataset's episodes."""

    def __init__(self, dataset_key: str):
        cfg = DATASETS[dataset_key]
        self.dataset_key = dataset_key
        self.repo_id = cfg["repo_id"]
        self.video_key = cfg["video_key"]
        self.local_dir = LOCAL_ROOT / cfg["local_dir_name"]

        info_path = hf_hub_download(self.repo_id, repo_type="dataset", filename="meta/info.json", local_dir=self.local_dir)
        self.info = json.loads(Path(info_path).read_text())
        episodes_path = hf_hub_download(self.repo_id, repo_type="dataset", filename="meta/episodes.jsonl", local_dir=self.local_dir)
        self.episodes_meta = [json.loads(line) for line in Path(episodes_path).read_text().splitlines()]
        self.chunks_size = self.info["chunks_size"]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes_meta)

    def load_episode(self, ep_idx: int):
        chunk = ep_idx // self.chunks_size
        parquet_path = hf_hub_download(
            self.repo_id, repo_type="dataset",
            filename=f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet",
            local_dir=self.local_dir,
        )
        video_path = hf_hub_download(
            self.repo_id, repo_type="dataset",
            filename=f"videos/chunk-{chunk:03d}/{self.video_key}/episode_{ep_idx:06d}.mp4",
            local_dir=self.local_dir,
        )

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        state = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        action = np.array(table.column("action").to_pylist(), dtype=np.float32)

        container = av.open(video_path)
        frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
        container.close()

        task = self.episodes_meta[ep_idx]["tasks"][0]
        return state, action, frames, task


def label_episode(model, source: SimplerEnvEpisodeSource, ep_idx: int, horizon_length: int, num_candidates: int) -> dict | None:
    state, action, frames, task = source.load_episode(ep_idx)
    n = state.shape[0]
    if n <= horizon_length:
        return None

    reward = _subgoal_rewards(state, source.dataset_key)

    embed_dim = None
    embeds = None
    candidates = None
    for t in range(n):
        state_t = state[t : t + 1]
        out = predict_action_candidates(model, [[frames[t]]], [task], state=state_t, num_samples=num_candidates)
        embed_t = out["embodied_action_tokens"].mean(axis=1)[0]
        cand_t = out["normalized_actions"]

        if embeds is None:
            embed_dim = embed_t.shape[0]
            embeds = np.zeros((n, embed_dim), dtype=np.float32)
            candidates = np.zeros((n, *cand_t.shape), dtype=np.float32)
        embeds[t] = embed_t
        candidates[t] = cand_t

    return {
        "state": state,
        "action": action,
        "reward": reward,
        "embed": embeds,
        "candidates": candidates,
        "horizon_length": horizon_length,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--bridge-episodes", type=int, default=500)
    p.add_argument("--fractal-episodes", type=int, default=500)
    p.add_argument("--horizon-length", type=int, default=5)
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--cuda", type=int, default=0)
    args = p.parse_args()

    model = baseframework.from_pretrained(args.ckpt_path).to(f"cuda:{args.cuda}").eval()

    out_path = Path(args.output_path)
    if out_path.exists():
        cache = dict(np.load(out_path, allow_pickle=True))
        done_keys = set(cache.get("_done_keys", np.array([], dtype=object)).tolist())
        print(f"Resuming from {out_path}: {len(done_keys)} episodes already done.")
    else:
        cache = {}
        done_keys = set()

    def save_checkpoint():
        cache["_done_keys"] = np.array(sorted(done_keys), dtype=object)
        cache["_horizon_length"] = args.horizon_length
        cache["_num_candidates"] = args.num_candidates
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **cache)
        tmp_path.replace(out_path)

    targets = [("bridge", args.bridge_episodes), ("fractal", args.fractal_episodes)]
    for dataset_key, num_episodes in targets:
        source = SimplerEnvEpisodeSource(dataset_key)
        n = min(num_episodes, source.num_episodes)
        print(f"Labeling {n}/{source.num_episodes} episodes from {dataset_key}")

        import tqdm
        for ep_idx in tqdm.tqdm(range(n), desc=dataset_key):
            key = f"{dataset_key}_{ep_idx}"
            if key in done_keys:
                continue
            result = label_episode(model, source, ep_idx, args.horizon_length, args.num_candidates)
            if result is None:
                continue
            for k, v in result.items():
                cache[f"{k}_{key}"] = v
            done_keys.add(key)

            if len(done_keys) % args.checkpoint_every == 0:
                save_checkpoint()

    save_checkpoint()
    print(f"Wrote {len(done_keys)} labeled episodes to {out_path}")


if __name__ == "__main__":
    main()
