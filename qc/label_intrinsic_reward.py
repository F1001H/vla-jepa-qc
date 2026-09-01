"""Adds VLA-JEPA-derived intrinsic (world-model prediction-error) reward on
top of an existing qc/label_rewards.py cache's subgoal reward -- mirrors
openpi's own layering (scripts/qc_label_rewards.py computes JEPA intrinsic
reward; scripts/add_subgoal_reward_to_qc_cache.py then adds a sparse subgoal
bonus on top of an already-intrinsic-labeled cache). Here we did it in the
opposite order for practical reasons (subgoal reward was cheap and fast to
validate first), but the final composition is the same: total reward =
intrinsic + subgoal.

Does NOT touch state/action/embed/candidates -- only replaces the reward
array, so the existing critic-training dataset code needs no changes.

Mechanics (verified against the real checkpoint before writing this):
  - VLA-JEPA's world model predicts each of T-1 temporal steps' tokens from
    the PREVIOUS step (a causal, teacher-forced next-step-tokens objective
    over an 8-frame/4-temporal-step clip), conditioned on a SEPARATE set of
    special tokens (`<|action_i|>`, extracted via the same Qwen forward
    pass predict_action_candidates already does for embodied_action_tokens
    -- pulling both out of one forward is free).
  - Reward = L1 error on the LAST predicted temporal step (the model's
    surprise at the most-recently-observed frame given the preceding
    window + action-token conditioning) -- the closest analog to openpi's
    single-step compute_intrinsic_reward, just computed from a longer
    causal context window instead of one prior frame.
  - Frames without enough history (t < num_frames - 1, i.e. the first 7
    frames of every episode) get intrinsic_reward = 0 -- can't be computed,
    and 0 is a neutral value consistent with the existing sparse-reward
    convention (rather than skipping these frames' transitions entirely,
    which would touch the critic-training dataset's indexing logic).

Usage:
    PYTHONPATH=. .venv/bin/python qc/label_intrinsic_reward.py \
        --dataset-root /path/to/physical-intelligence-libero \
        --ckpt-path /path/to/VLA-JEPA-LIBERO.pt \
        --input-cache /path/to/qc_cache_full.npz \
        --output-path qc_cache_full_intrinsic.npz \
        [--max-episodes N]
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

from starVLA.model.framework.base_framework import baseframework


def _decode_image(cell: dict) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


class ParquetLiberoDataset:
    """Same as qc/label_rewards.py's -- duplicated here (not imported) since
    that module also imports torch/model-loading code we don't need twice,
    and this keeps the two labeling scripts independently runnable."""

    def __init__(self, root: str):
        self.root = Path(root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.chunks_size = info["chunks_size"]
        self.data_path_template = info["data_path"]
        self.episodes = [json.loads(line) for line in (self.root / "meta" / "episodes.jsonl").read_text().splitlines()]
        self.num_episodes = len(self.episodes)

    def load_episode(self, ep_idx: int):
        chunk = ep_idx // self.chunks_size
        path = self.root / self.data_path_template.format(episode_chunk=chunk, episode_index=ep_idx)
        rows = pq.read_table(path).to_pylist()
        images = [r["image"] for r in rows]
        wrist_images = [r["wrist_image"] for r in rows]
        task = self.episodes[ep_idx]["tasks"][0]
        return images, wrist_images, task


@torch.no_grad()
def _intrinsic_reward_at(model, images: list, wrist_images: list, task: str, t: int, num_frames: int) -> float:
    """images/wrist_images: full per-episode decoded PIL image lists.
    t: frame index to compute reward for (needs t >= num_frames - 1)."""
    img_t = images[t]
    wrist_t = wrist_images[t]

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[[img_t, wrist_t]],
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
    primary_window = [np.asarray(images[window_start + i]) for i in range(num_frames)]
    wrist_window = [np.asarray(wrist_images[window_start + i]) for i in range(num_frames)]
    batch_videos = np.stack([np.stack(primary_window), np.stack(wrist_window)])[None]  # [1, V=2, T, H, W, 3]
    batch_videos = batch_videos.transpose(0, 1, 2, 5, 3, 4)  # [1, V, T, 3, H, W]
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
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--input-cache", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Write partial progress to --output-path every N episodes, so a crash (this "
             "machine has had power issues mid-session) loses at most N episodes of work, "
             "not the whole multi-hour run. Overwrites --output-path in place each time.",
    )
    args = p.parse_args()

    model = baseframework.from_pretrained(args.ckpt_path).to(f"cuda:{args.cuda}").eval()
    num_frames = model.config.framework.vj2_model.num_frames

    ds = ParquetLiberoDataset(args.dataset_root)

    # Resume support: if --output-path already exists (from a previous,
    # interrupted run of THIS script), it already has some episodes' rewards
    # combined -- load it instead of --input-cache and skip those episodes.
    # _intrinsic_done_episodes tracks which ones, since the reward array
    # itself doesn't distinguish "not yet processed" from "processed but
    # happened to get 0 intrinsic reward everywhere" (boundary frames legitimately
    # do that already, so we can't infer completion from the values alone).
    out_path = Path(args.output_path)
    if out_path.exists():
        cache = dict(np.load(out_path))
        done_episodes = set(cache.get("_intrinsic_done_episodes", np.array([], dtype=np.int64)).tolist())
        print(f"Resuming from {out_path}: {len(done_episodes)} episodes already done.")
    else:
        cache = dict(np.load(args.input_cache))
        done_episodes = set()

    n_ep = int(cache["_num_episodes"])
    num_episodes = n_ep if args.max_episodes is None else min(args.max_episodes, n_ep)
    print(f"Labeling intrinsic reward for {num_episodes}/{n_ep} episodes, num_frames={num_frames}")

    def save_checkpoint():
        cache["_intrinsic_reward_added"] = True
        cache["_intrinsic_num_frames"] = num_frames
        cache["_intrinsic_done_episodes"] = np.array(sorted(done_episodes), dtype=np.int64)
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **cache)
        tmp_path.replace(out_path)  # atomic on the same filesystem -- never leaves a half-written output file

    for ep_idx in tqdm.tqdm(range(num_episodes)):
        if ep_idx in done_episodes:
            continue
        if f"state_{ep_idx}" not in cache:
            continue  # episode was skipped (too short) by label_rewards.py
        raw_images, raw_wrist_images, task = ds.load_episode(ep_idx)
        images = [_decode_image(im) for im in raw_images]
        wrist_images = [_decode_image(im) for im in raw_wrist_images]
        n = len(images)
        subgoal_reward = cache[f"reward_{ep_idx}"]
        total_reward = subgoal_reward.copy()
        for t in range(num_frames - 1, n):
            intrinsic = _intrinsic_reward_at(model, images, wrist_images, task, t, num_frames)
            total_reward[t] = subgoal_reward[t] + intrinsic
        cache[f"reward_{ep_idx}"] = total_reward
        done_episodes.add(ep_idx)

        if len(done_episodes) % args.checkpoint_every == 0:
            save_checkpoint()

    save_checkpoint()
    print(f"Wrote combined (subgoal + intrinsic) reward cache to {args.output_path}")


if __name__ == "__main__":
    main()
