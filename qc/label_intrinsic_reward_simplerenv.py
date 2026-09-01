"""Adds VLA-JEPA-derived intrinsic (world-model prediction-error) reward on
top of an existing qc/label_rewards_simplerenv.py cache's subgoal reward --
SimplerEnv counterpart of qc/label_intrinsic_reward.py (LIBERO). Same
mechanics, same reward composition (total = intrinsic + subgoal); only the
data source and camera count differ.

Does NOT touch state/action/embed/candidates -- only replaces the reward_*
arrays, so the existing critic-training dataset code needs no changes.

Differences from the LIBERO version:
  - SINGLE camera (no wrist), matching qc/label_rewards_simplerenv.py's
    confirmed single-image SimplerEnv inference. The world model still
    expects V=2 though (matches its LIBERO primary+wrist training setup) --
    per the VLA-JEPA paper: "when fewer than two camera views are
    available, we duplicate the world-state representation and concatenate
    the two copies." So: encode the single camera window once (V=1), then
    duplicate that embedding to V=2 before the final concatenation, rather
    than stacking real 2 camera views like LIBERO does. This was flagged as
    an unverified assumption in an earlier version of this file; the paper
    confirms it directly, so it's no longer a guess -- but the SPECIFIC
    mechanism (embedding-level duplication vs. e.g. duplicating raw pixels
    before encoding) is still an inference from that one sentence, not
    something the paper spells out at that level of detail. Still worth a
    sanity check against real intrinsic-reward magnitudes on first run.
  - Reads episodes via qc/label_rewards_simplerenv.py's
    SimplerEnvEpisodeSource (duplicated here rather than imported, same
    reasoning as the LIBERO script: keeps this independently runnable and
    avoids pulling in qc/sampling.py's candidate-generation dependencies
    this script doesn't need) -- video-decoded (PyAV) frames, not
    embedded-PNG parquet cells.
  - Iterates over the base cache's `_done_keys` (format "{dataset}_{idx}",
    e.g. "bridge_42") rather than re-deriving episode counts, so it exactly
    matches whatever label_rewards_simplerenv.py actually labeled
    (including episodes it skipped for being too short).
  - Episodes are much shorter here (25-115 frames vs LIBERO's 75-505), so a
    larger fraction of each episode falls in the "not enough history yet"
    zero-intrinsic-reward window (first num_frames-1 frames) -- expected,
    not a bug.

Usage:
    PYTHONPATH=. .venv/bin/python qc/label_intrinsic_reward_simplerenv.py \
        --ckpt-path /path/to/VLA-JEPA-SimplerEnv.pt \
        --input-cache /path/to/qc_cache_simplerenv.npz \
        --output-path qc_cache_simplerenv_intrinsic.npz \
        [--checkpoint-every 25]
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from huggingface_hub import hf_hub_download
from PIL import Image

from starVLA.model.framework.base_framework import baseframework

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


class SimplerEnvEpisodeSource:
    """Duplicated from qc/label_rewards_simplerenv.py -- see module
    docstring for why. Only the parts this script needs (frames + task)."""

    def __init__(self, dataset_key: str):
        cfg = DATASETS[dataset_key]
        self.repo_id = cfg["repo_id"]
        self.video_key = cfg["video_key"]
        self.local_dir = LOCAL_ROOT / cfg["local_dir_name"]

        info_path = hf_hub_download(self.repo_id, repo_type="dataset", filename="meta/info.json", local_dir=self.local_dir)
        self.info = json.loads(Path(info_path).read_text())
        episodes_path = hf_hub_download(self.repo_id, repo_type="dataset", filename="meta/episodes.jsonl", local_dir=self.local_dir)
        self.episodes_meta = [json.loads(line) for line in Path(episodes_path).read_text().splitlines()]
        self.chunks_size = self.info["chunks_size"]

    def load_episode(self, ep_idx: int):
        chunk = ep_idx // self.chunks_size
        video_path = hf_hub_download(
            self.repo_id, repo_type="dataset",
            filename=f"videos/chunk-{chunk:03d}/{self.video_key}/episode_{ep_idx:06d}.mp4",
            local_dir=self.local_dir,
        )
        container = av.open(video_path)
        frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
        container.close()
        task = self.episodes_meta[ep_idx]["tasks"][0]
        return frames, task


@torch.no_grad()
def _intrinsic_reward_at(model, frames: list, task: str, t: int, num_frames: int) -> float:
    """frames: full per-episode decoded PIL image list (single camera).
    t: frame index to compute reward for (needs t >= num_frames - 1)."""
    img_t = frames[t]

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[[img_t]],
        instructions=[task],
        prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt},
    )
    action_indices = torch.isin(
        qwen_inputs["input_ids"], torch.tensor(model.action_token_ids, device=qwen_inputs["input_ids"].device)
    ).nonzero(as_tuple=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]
        B, _, H = last_hidden.shape
        action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)

    window_start = t - num_frames + 1
    primary_window = [np.asarray(frames[window_start + i]) for i in range(num_frames)]
    # Single camera -> encode as V=1, then duplicate the resulting world-state
    # representation to V=2 before concatenating -- per the VLA-JEPA paper:
    # "when fewer than two camera views are available, we duplicate the
    # world-state representation and concatenate the two copies." (Encoding
    # once and duplicating the embedding, rather than encoding the same
    # frames twice through vj_encoder, is equivalent and cheaper.)
    batch_videos = np.stack([np.stack(primary_window)])[None]  # [1, V=1, T, H, W, 3]
    batch_videos = batch_videos.transpose(0, 1, 2, 5, 3, 4)  # [1, V=1, T, 3, H, W]
    B_, V, T, C, Himg, Wimg = batch_videos.shape
    batch_videos_flat = batch_videos.reshape(B_ * V, T, C, Himg, Wimg)
    input_videos = torch.cat(
        [model.vj_processor(videos=batch_videos_flat[i], return_tensors="pt")["pixel_values_videos"].to(model.vj_encoder.device)
         for i in range(B_ * V)],
        dim=0,
    )

    video_embeddings = model.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
    num_temporal_steps = T // model.vj_encoder.config.tubelet_size
    tokens_per_clip, embed_dim = video_embeddings.shape[1:]
    tokens_per_step = tokens_per_clip // num_temporal_steps
    video_embeddings = video_embeddings.reshape(B_, V, num_temporal_steps, tokens_per_step, embed_dim)
    video_embeddings = torch.cat([video_embeddings, video_embeddings], dim=1)  # V=1 -> V=2, duplicated copy
    V = 2
    video_embeddings = video_embeddings.permute(0, 2, 3, 1, 4).contiguous().reshape(
        B_, num_temporal_steps * tokens_per_step, V * embed_dim
    )

    Tsteps = num_temporal_steps
    input_states = video_embeddings[:, : video_embeddings.shape[1] // Tsteps * (Tsteps - 1), :]
    gt_states = video_embeddings[:, video_embeddings.shape[1] // Tsteps :, :]

    predicted_states = model.vj_predictor(input_states.float(), action_tokens.float())
    error = F.l1_loss(predicted_states, gt_states, reduction="none")
    last_step_error = error.view(B_, Tsteps - 1, tokens_per_step, -1)[:, -1, :, :].mean()
    return float(last_step_error.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--input-cache", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Write partial progress to --output-path every N episodes, so a crash "
             "loses at most N episodes of work, not the whole run. Overwrites "
             "--output-path in place each time.",
    )
    args = p.parse_args()

    model = baseframework.from_pretrained(args.ckpt_path).to(f"cuda:{args.cuda}").eval()
    num_frames = model.config.framework.vj2_model.num_frames

    # Resume support: if --output-path already exists (from a previous,
    # interrupted run of THIS script), load it instead of --input-cache and
    # skip its already-done keys. _intrinsic_done_keys tracks which ones,
    # mirroring label_rewards_simplerenv.py's own _done_keys convention.
    out_path = Path(args.output_path)
    if out_path.exists():
        cache = dict(np.load(out_path, allow_pickle=True))
        done_keys = set(cache.get("_intrinsic_done_keys", np.array([], dtype=object)).tolist())
        print(f"Resuming from {out_path}: {len(done_keys)} episodes already done.")
    else:
        cache = dict(np.load(args.input_cache, allow_pickle=True))
        done_keys = set()

    base_keys = sorted(cache["_done_keys"].tolist())
    print(f"Labeling intrinsic reward for {len(base_keys)} episodes, num_frames={num_frames}")

    def save_checkpoint():
        cache["_intrinsic_reward_added"] = True
        cache["_intrinsic_num_frames"] = num_frames
        cache["_intrinsic_done_keys"] = np.array(sorted(done_keys), dtype=object)
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **cache)
        tmp_path.replace(out_path)  # atomic on the same filesystem -- never leaves a half-written output file

    sources = {}
    for key in tqdm.tqdm(base_keys):
        if key in done_keys:
            continue
        if f"reward_{key}" not in cache:
            continue  # episode was skipped (too short) by label_rewards_simplerenv.py
        dataset_key, ep_idx_str = key.rsplit("_", 1)
        ep_idx = int(ep_idx_str)
        if dataset_key not in sources:
            sources[dataset_key] = SimplerEnvEpisodeSource(dataset_key)
        frames, task = sources[dataset_key].load_episode(ep_idx)
        n = len(frames)
        subgoal_reward = cache[f"reward_{key}"]
        total_reward = subgoal_reward.copy()
        for t in range(num_frames - 1, n):
            intrinsic = _intrinsic_reward_at(model, frames, task, t, num_frames)
            total_reward[t] = subgoal_reward[t] + intrinsic
        cache[f"reward_{key}"] = total_reward
        done_keys.add(key)

        if len(done_keys) % args.checkpoint_every == 0:
            save_checkpoint()

    save_checkpoint()
    print(f"Wrote combined (subgoal + intrinsic) reward cache to {args.output_path}")


if __name__ == "__main__":
    main()
