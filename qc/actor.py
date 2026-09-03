"""Best-of-N critic-scored action selection for VLA-JEPA, mirroring openpi's
src/qc/actor.py (same score formula and argmin-by-default rationale -- a
frozen critic scoring predicted error/novelty should be MINIMIZED at eval
time, not maximized, since there's no further learning payoff to picking a
riskier candidate; see that file's docstring for the full argument).

Operates on ONE live observation per call, since that's how
deployment/model_server/server_policy.py's inference loop calls
predict_action -- one env step at a time, not a training batch.
"""

import numpy as np
import torch

from qc.sampling import predict_action_candidates


def best_of_n_action(
    model,
    critic,
    batch_images,
    instructions,
    state=None,
    num_samples: int = 8,
    horizon_length: int = 5,
    *,
    q_agg: str = "mean",
    uncertainty_penalty: float = 0.0,
    actor_disagreement_penalty: float = 0.0,
    critic_weight: float = 1.0,
    maximize_score: bool = False,
    selection_mode: str = "score",
    normalize_score_terms: bool = False,
    rank_score_terms: bool = False,
    candidate_temperature: float = 1.0,
    candidate_sde_noise_scale: float = 0.0,
    **kwargs,
) -> dict:
    """Returns dict shaped like predict_action's original output (single
    winning candidate, NOT a batch of N) so the server/client wire format
    doesn't change: {"normalized_actions": [chunk_len, action_dim], ...}.

    horizon_length must match the critic's training horizon_length (default
    5, see qc/train_critic.py) -- the critic only ever saw the first
    horizon_length steps of a chunk, NOT the model's native chunk_len (7,
    future_action_window_size+1), so scoring must truncate to match. The
    FULL native-length candidate is still what gets returned/executed --
    only the critic's *scoring* input is truncated (same asymmetry as
    label_rewards.py's cache: candidates stored at native length, chunk
    aggregation/truncation happens at read/use time).
    """
    out = predict_action_candidates(
        model, batch_images, instructions, state=state, num_samples=num_samples,
        temperature=candidate_temperature, sde_noise_scale=candidate_sde_noise_scale,
    )
    candidates = out["normalized_actions"]  # [N, chunk_len, action_dim] np -- native chunk_len (e.g. 7)
    embodied_action_tokens = out["embodied_action_tokens"]  # [1, num_tokens, H] np

    device = next(critic.parameters()).device
    dtype = next(critic.parameters()).dtype

    embed = embodied_action_tokens.mean(axis=1)  # [1, H] pooled -- one per obs, not per-candidate
    embed_t = torch.from_numpy(embed).to(device=device, dtype=dtype).repeat(num_samples, 1)  # [N, H]

    proprio = np.asarray(state, dtype=np.float32).reshape(1, -1) if state is not None else np.zeros((1, 0), dtype=np.float32)
    proprio_t = torch.from_numpy(proprio).to(device=device, dtype=dtype).repeat(num_samples, 1)  # [N, proprio_dim]

    candidates_t = torch.from_numpy(candidates).to(device=device, dtype=dtype)  # [N, chunk_len, action_dim]
    candidates_for_critic = candidates_t[:, :horizon_length, :]  # [N, horizon_length, action_dim] -- matches critic's training input dim

    # Actor-side disagreement: how much do the N candidates disagree with
    # EACH OTHER, independent of the critic -- free, no extra forward pass.
    # Computed over the same truncated window the critic sees, for consistency.
    action_mean = candidates_for_critic.mean(dim=0, keepdim=True)  # [1, horizon_length, action_dim]
    actor_disagreement = torch.sqrt(((candidates_for_critic - action_mean) ** 2).mean(dim=(1, 2)))  # [N]

    with torch.no_grad():
        qs = critic(embed_t, proprio_t, candidates_for_critic)  # [num_qs, N]
    q = qs.min(dim=0).values if q_agg == "min" else qs.mean(dim=0)  # [N]
    disagreement = qs.std(dim=0)  # [N] -- cross-head disagreement, critic's own epistemic uncertainty proxy

    if rank_score_terms:
        # Alternative to z-scoring: convert each term to its zero-centered
        # RANK among the N candidates instead of a rescaled value. Motivated
        # by an empirical A/B (2026-09-03, local): normalize_score_terms
        # (z-score) made libero_10 and LIBERO-Plus's Camera Viewpoints WORSE,
        # not better, despite the scale-mismatch diagnosis being solid --
        # plausible explanation is the critic's raw q/disagreement spread is
        # so tiny (near noise floor, see actor's z-score comment below) that
        # z-scoring promotes that noise to unit variance, giving it equal
        # say to actor_disagreement's (probably more real) signal. Rank
        # transform is robust to this: a nearly-random term's RANK is still
        # bounded, symmetric noise instead of being rescaled up to dominate.
        def _rank(x: torch.Tensor) -> torch.Tensor:
            order = torch.argsort(x)
            ranks = torch.empty_like(order, dtype=x.dtype)
            ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
            return ranks - (len(x) - 1) / 2.0

        q = _rank(q)
        disagreement = _rank(disagreement)
        actor_disagreement = _rank(actor_disagreement)
    elif normalize_score_terms:
        # Raw q/disagreement/actor_disagreement live on wildly different scales
        # (diagnosed offline against qc_cache_full_intrinsic.npz: median
        # within-timestep spread ratio q:disagreement:actor_disagreement was
        # roughly 4:1:21) -- summing them raw means actor_disagreement
        # silently dominates selection regardless of the penalty weights,
        # since it's the SPREAD across the N candidates (not raw magnitude)
        # that determines argmax/argmin. Z-score each term across the N
        # candidates (this call's own batch) so the weights actually control
        # relative influence. NOTE (2026-09-03): empirically this made
        # results WORSE on non-saturated suites, not better -- see
        # rank_score_terms's comment above for the leading hypothesis why.
        def _zscore(x: torch.Tensor) -> torch.Tensor:
            return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)

        q = _zscore(q)
        disagreement = _zscore(disagreement)
        actor_disagreement = _zscore(actor_disagreement)

    score = (
        critic_weight * q
        + uncertainty_penalty * disagreement
        + actor_disagreement_penalty * actor_disagreement
    )

    if selection_mode == "random":
        # Control: ignores q/disagreement/actor_disagreement entirely, picks
        # uniformly among the N candidates. Establishes the floor -- if this
        # matches "best ablation" performance, the apparent best-of-N gains
        # throughout this investigation may just be free sample diversity
        # from stochastic decoding, not selection quality.
        best_idx = int(torch.randint(0, num_samples, (1,)).item())
    elif selection_mode == "majority_vote":
        per_head_best = torch.argmax(qs, dim=-1) if maximize_score else torch.argmin(qs, dim=-1)  # [num_qs]
        votes = torch.bincount(per_head_best, minlength=num_samples)
        best_idx = int(torch.argmax(votes).item())
    else:
        best_idx = int((torch.argmax(score) if maximize_score else torch.argmin(score)).item())

    return {
        # Keep the leading batch dim of 1 -- predict_action's original contract
        # is [B, chunk_len, action_dim], and model2libero_interface.py's step()
        # unconditionally does normalized_actions[0] to strip it. Returning
        # candidates[best_idx] bare (no batch dim) made that [0] silently
        # collapse the CHUNK dim instead, producing a 1D array and crashing
        # unnormalize_actions's [:, 6] indexing two calls later.
        "normalized_actions": candidates[best_idx : best_idx + 1],  # [1, chunk_len, action_dim]
        "embodied_action_tokens": embodied_action_tokens,
    }
