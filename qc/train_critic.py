"""PyTorch port of openpi's src/qc/train_step.py -- chunked-TD critic
training (Li/Zhou/Levine, NeurIPS 2025, https://github.com/ColinQiyangLi/qc),
same algorithm (target network EMA, n-step chunked Bellman target, best-of-N
target selection over cached next-state candidates), operating on the cache
produced by qc/label_rewards.py instead of openpi's JAX cache.

Never loads the VLA-JEPA backbone: everything the critic needs (embed,
proprio, action_chunk, reward, next-state embed/candidates) was precomputed
once by label_rewards.py, exactly the decoupling openpi's own critic training
relies on (see that file's module docstring).
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import tqdm

from qc.critic import QChunkCritic


class QChunkCacheDataset(torch.utils.data.Dataset):
    """Reads label_rewards.py's .npz cache and builds (t, t+horizon_length)
    transitions with an n-step discounted-sum reward over the chunk (the
    whole chunk is treated as one macro-action for a single Bellman backup,
    matching train_step.py's single `discount**horizon_length` factor)."""

    def __init__(self, cache_path: str, discount: float):
        data = np.load(cache_path)
        self.horizon_length = int(data["_horizon_length"])
        num_episodes = int(data["_num_episodes"])
        self.discount = discount

        self.index = []  # (ep_key_prefix, t)
        self.episodes = {}
        for ep_idx in range(num_episodes):
            key = f"state_{ep_idx}"
            if key not in data:
                continue  # skipped episode (too short), see label_rewards.py
            state = data[f"state_{ep_idx}"]
            action = data[f"action_{ep_idx}"]
            reward = data[f"reward_{ep_idx}"]
            embed = data[f"embed_{ep_idx}"]
            candidates = data[f"candidates_{ep_idx}"]
            self.episodes[ep_idx] = (state, action, reward, embed, candidates)
            n = state.shape[0]
            for t in range(n - self.horizon_length):
                self.index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.index[idx]
        state, action, reward, embed, candidates = self.episodes[ep_idx]
        h = self.horizon_length
        t1 = t + h

        chunk_reward = sum(self.discount**k * reward[t + k] for k in range(h))

        return {
            "embed_t": embed[t],
            "proprio_t": state[t],
            "action_chunk": action[t:t1],
            "reward": np.float32(chunk_reward),
            "embed_th": embed[t1],
            "proprio_th": state[t1],
            # candidates were cached at the MODEL's native chunk length (7,
            # from future_action_window_size+1), not the training
            # horizon_length (5) -- truncate to match action_chunk's length,
            # same as openpi's own src/qc/actor.py does
            # (candidates[:, :horizon_length, :action_dim]) before scoring.
            "next_action_candidates": candidates[t1][:, :h, :],  # [num_candidates, h, action_dim]
            "mask": np.float32(1.0),  # t1 always valid by construction (t ranges over n - h)
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in batch[0]}


def train(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    ds = QChunkCacheDataset(args.cache_path, args.discount)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=True,
    )
    print(f"{len(ds)} transitions from cache {args.cache_path}")

    sample = ds[0]
    embed_dim = sample["embed_t"].shape[0]
    proprio_dim = sample["proprio_t"].shape[0]
    action_dim = sample["action_chunk"].shape[-1]
    horizon_length = ds.horizon_length

    critic = QChunkCritic(
        embed_dim, proprio_dim, action_dim, horizon_length,
        hidden_dims=tuple(args.hidden_dims), num_qs=args.num_qs, layer_norm=True,
    ).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device).float() for k, v in batch.items()}
            embed_t, proprio_t, action_chunk = batch["embed_t"], batch["proprio_t"], batch["action_chunk"]
            reward, mask = batch["reward"], batch["mask"]
            embed_th, proprio_th = batch["embed_th"], batch["proprio_th"]
            next_action_candidates = batch["next_action_candidates"]  # [B, N, h, a]

            batch_size, num_candidates = next_action_candidates.shape[:2]

            with torch.no_grad():
                embed_th_tiled = embed_th.repeat_interleave(num_candidates, dim=0)
                proprio_th_tiled = proprio_th.repeat_interleave(num_candidates, dim=0)
                candidates_flat = next_action_candidates.reshape(
                    batch_size * num_candidates, *next_action_candidates.shape[2:]
                )
                next_qs = target_critic(embed_th_tiled, proprio_th_tiled, candidates_flat)  # [num_qs, B*N]
                next_q_per_cand = next_qs.min(dim=0).values if args.q_agg == "min" else next_qs.mean(dim=0)
                next_q_per_cand = next_q_per_cand.reshape(batch_size, num_candidates)
                next_q = next_q_per_cand.max(dim=1).values  # [B] best candidate per example

                target_q = reward + (args.discount**horizon_length) * mask * next_q

            qs = critic(embed_t, proprio_t, action_chunk)  # [num_qs, B]
            loss = (((qs - target_q.unsqueeze(0)) ** 2) * mask.unsqueeze(0)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(1 - args.tau).add_(args.tau * p.data)

            step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=loss.item(), q_mean=qs.mean().item())

        torch.save(critic.state_dict(), args.output_path)
        print(f"Saved checkpoint to {args.output_path} after epoch {epoch}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--q-agg", type=str, default="mean", choices=["mean", "min"])
    p.add_argument("--num-qs", type=int, default=5)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512, 512])
    p.add_argument("--cuda", type=int, default=0)
    train(p.parse_args())
