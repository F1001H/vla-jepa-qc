"""Trains DuelingQChunkCritic (qc/critic.py) instead of the monolithic
QChunkCritic -- see that class's docstring for the motivation (the
monolithic critic converges cleanly but has no gradient incentive to
resolve the tiny action-dependent residual that best-of-N selection
actually depends on).

Adapted from qc/train_critic.py: same cache format, same chunked n-step TD
setup, same target-network EMA -- the only real difference is the TD
target/query Q are now computed via DuelingQChunkCritic.combine() using a
mean-centered advantage instead of a single monolithic forward pass, which
requires ALSO reading each transition's OWN-timestep candidates (not just
the next-timestep ones qc/train_critic.py's dataset class reads) as the
centering baseline for the query Q.
"""

import argparse
import copy

import numpy as np
import torch
import tqdm

from qc.critic import DuelingQChunkCritic


class QChunkCacheDatasetWithCurrentCandidates(torch.utils.data.Dataset):
    """Same as qc/train_critic.py's QChunkCacheDataset, but ALSO returns
    candidates[t] (the acted-upon timestep's own alternatives, used as the
    dueling centering baseline for the query Q) alongside
    next_action_candidates (candidates[t1], used for the TD target as
    before)."""

    def __init__(self, cache_path: str, discount: float):
        data = np.load(cache_path, allow_pickle=True)
        self.horizon_length = int(data["_horizon_length"])
        self.discount = discount

        if "_done_keys" in data:
            episode_keys = sorted(data["_done_keys"].tolist())
        else:
            episode_keys = [str(i) for i in range(int(data["_num_episodes"]))]

        self.index = []
        self.episodes = {}
        for key in episode_keys:
            if f"state_{key}" not in data:
                continue
            state = data[f"state_{key}"]
            action = data[f"action_{key}"]
            reward = data[f"reward_{key}"]
            embed = data[f"embed_{key}"]
            candidates = data[f"candidates_{key}"]
            self.episodes[key] = (state, action, reward, embed, candidates)
            n = state.shape[0]
            for t in range(n - self.horizon_length):
                self.index.append((key, t))

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
            "current_action_candidates": candidates[t][:, :h, :],  # NEW vs train_critic.py -- dueling baseline for the query Q
            "reward": np.float32(chunk_reward),
            "embed_th": embed[t1],
            "proprio_th": state[t1],
            "next_action_candidates": candidates[t1][:, :h, :],
            "mask": np.float32(1.0),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in batch[0]}


def score_candidates_dueling(critic, embed, proprio, candidates, q_agg: str):
    """embed/proprio: [B, ...], candidates: [B, N, h, action_dim].
    Returns per-example best-candidate dueling Q, aggregated over the
    ensemble via q_agg, maxed over the N candidates -- same shape/role as
    train_critic.py's next_q computation, just using the dueling combine.
    Also returns the raw advantage [num_qs, B, N] for reuse as a centering
    baseline elsewhere (score_query_dueling uses this same helper's
    intermediate values for the CURRENT timestep instead of just the max).
    """
    batch_size, num_candidates = candidates.shape[:2]
    embed_tiled = embed.repeat_interleave(num_candidates, dim=0)
    proprio_tiled = proprio.repeat_interleave(num_candidates, dim=0)
    candidates_flat = candidates.reshape(batch_size * num_candidates, *candidates.shape[2:])

    v = critic.forward_v(embed, proprio)  # [num_qs, B]
    a_flat = critic.forward_a(embed_tiled, proprio_tiled, candidates_flat)  # [num_qs, B*N]
    a = a_flat.reshape(critic.num_qs, batch_size, num_candidates)  # [num_qs, B, N]

    # NOTE: DuelingQChunkCritic.combine() is only correct for the 2D
    # "single query action, N-candidate baseline" shape (v/a: [num_qs,B],
    # baseline: [num_qs,B]) -- this is the 3D "score all N candidates at
    # once" case (a: [num_qs,B,N]), which needs keepdim=True so the
    # baseline broadcasts against N, not B. Verified this distinction with
    # a standalone shape check before trusting it (combine() without
    # keepdim raises a RuntimeError here, doesn't silently misbehave).
    baseline = a.mean(dim=-1, keepdim=True)  # [num_qs, B, 1]
    q_candidates = v.unsqueeze(-1) + (a - baseline)  # [num_qs, B, N]
    q_per_cand = q_candidates.min(dim=0).values if q_agg == "min" else q_candidates.mean(dim=0)  # [B, N]
    return q_per_cand, v, a


def train(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    ds = QChunkCacheDatasetWithCurrentCandidates(args.cache_path, args.discount)
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

    critic = DuelingQChunkCritic(
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
            current_action_candidates = batch["current_action_candidates"]  # [B, N, h, a]
            next_action_candidates = batch["next_action_candidates"]  # [B, N, h, a]

            with torch.no_grad():
                # TD target: best dueling-Q candidate at the NEXT state, via the target network.
                next_q_per_cand, _, _ = score_candidates_dueling(
                    target_critic, embed_th, proprio_th, next_action_candidates, args.q_agg
                )
                next_q = next_q_per_cand.max(dim=1).values  # [B]
                target_q = reward + (args.discount**horizon_length) * mask * next_q

            # Query Q: the EXECUTED action_chunk's dueling-Q at the CURRENT
            # state, centered against that SAME state's own candidate pool
            # (current_action_candidates) -- this is the identifiability
            # constraint that forces real gradient onto A (see
            # DuelingQChunkCritic's docstring): V alone can't distinguish
            # the executed action from the alternatives, only the centered
            # advantage term can, so the TD loss can't be satisfied by
            # fitting V alone.
            v_t = critic.forward_v(embed_t, proprio_t)  # [num_qs, B]
            a_t = critic.forward_a(embed_t, proprio_t, action_chunk)  # [num_qs, B]
            batch_size, num_candidates = current_action_candidates.shape[:2]
            embed_t_tiled = embed_t.repeat_interleave(num_candidates, dim=0)
            proprio_t_tiled = proprio_t.repeat_interleave(num_candidates, dim=0)
            cand_flat = current_action_candidates.reshape(batch_size * num_candidates, *current_action_candidates.shape[2:])
            a_t_candidates_flat = critic.forward_a(embed_t_tiled, proprio_t_tiled, cand_flat)
            a_t_candidates = a_t_candidates_flat.reshape(critic.num_qs, batch_size, num_candidates)

            qs = DuelingQChunkCritic.combine(v_t, a_t, a_t_candidates)  # [num_qs, B]
            loss = (((qs - target_q.unsqueeze(0)) ** 2) * mask.unsqueeze(0)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(1 - args.tau).add_(args.tau * p.data)

            step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=loss.item(), q_mean=qs.mean().item(), a_mean=a_t.mean().item(), v_mean=v_t.mean().item())

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
