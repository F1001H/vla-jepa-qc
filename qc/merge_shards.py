"""Merges per-GPU shard caches produced by qc/label_rewards.py or
qc/label_intrinsic_reward.py (both support --shard-id/--num-shards for
parallelizing the labeling pass across multiple GPUs).

Two modes, matching the two scripts' different shard-output shapes:
  - subgoal: each shard file contains ONLY the per-episode keys for
    episodes it owns (ep_idx %% num_shards == shard_id) -- a plain union
    across shard files (keys are disjoint by construction).
  - intrinsic: each shard file is a FULL cache copy (every episode's
    state/action/embed/candidates present in every shard, inherited
    unmodified from the input cache) but only its OWN episodes' reward_*
    was actually updated with intrinsic reward -- so the merge picks each
    episode's reward_{ep_idx} from its owning shard specifically, and
    takes all other keys from any one shard (they're identical copies).

--shard-paths must be given in shard-id order (0, 1, 2, ... num_shards-1)
-- this script trusts that ordering, it doesn't parse shard-id back out of
filenames.

Usage:
    PYTHONPATH=. .venv/bin/python qc/merge_shards.py \
        --mode subgoal \
        --shard-paths qc_cache_shard0.npz qc_cache_shard1.npz ... \
        --output-path qc_cache_full.npz
"""

import argparse

import numpy as np


def merge_subgoal(shard_paths: list[str]) -> dict:
    merged = {}
    done_episodes = set()
    meta_keys = ("_num_episodes", "_horizon_length", "_num_candidates", "_ckpt_path")
    for path in shard_paths:
        shard = dict(np.load(path, allow_pickle=True))
        for k, v in shard.items():
            if k == "_done_episodes":
                done_episodes.update(v.tolist())
                continue
            if k in meta_keys:
                merged[k] = v  # identical across shards, last write wins
                continue
            if k in merged:
                raise ValueError(f"Key {k!r} present in multiple shards -- shards should own disjoint episodes")
            merged[k] = v
    merged["_done_episodes"] = np.array(sorted(done_episodes), dtype=np.int64)
    return merged


def merge_intrinsic(shard_paths: list[str], num_shards: int) -> dict:
    shards = [dict(np.load(path, allow_pickle=True)) for path in shard_paths]
    merged = dict(shards[0])  # base: any shard has the full state/action/embed/candidates set
    num_episodes = int(merged["_num_episodes"])
    for ep_idx in range(num_episodes):
        owner = ep_idx % num_shards
        key = f"reward_{ep_idx}"
        if key in shards[owner]:
            merged[key] = shards[owner][key]
    merged.pop("_intrinsic_done_episodes", None)  # per-shard bookkeeping, meaningless post-merge
    merged["_intrinsic_reward_added"] = True
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["subgoal", "intrinsic"], required=True)
    p.add_argument("--shard-paths", nargs="+", required=True, help="In shard-id order: shard0 shard1 ... shardN-1")
    p.add_argument("--output-path", required=True)
    args = p.parse_args()

    if args.mode == "subgoal":
        merged = merge_subgoal(args.shard_paths)
    else:
        merged = merge_intrinsic(args.shard_paths, len(args.shard_paths))

    np.savez(args.output_path, **merged)
    n_ep = int(merged["_num_episodes"])
    n_done = len(merged.get("_done_episodes", [])) if args.mode == "subgoal" else n_ep
    print(f"Merged {len(args.shard_paths)} shards ({args.mode}) -> {args.output_path} ({n_done}/{n_ep} episodes)")


if __name__ == "__main__":
    main()
