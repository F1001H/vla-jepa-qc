# QC (critic-scored best-of-N) variant of server_policy.py. Wraps a frozen
# VLA_JEPA checkpoint + a trained qc.critic.QChunkCritic behind the SAME
# predict_action(**payload) interface WebsocketPolicyServer._route_message
# dispatches to, so eval_libero.py / M1Inference need zero changes to talk
# to this server instead of the plain one.
#
# NOTE: --critic-checkpoint-path is required. There is deliberately no
# fallback to a randomly-initialized critic for a "real" eval run -- that
# would silently produce meaningless best-of-N selection. Use
# --allow-untrained-critic only to smoke-test the plumbing (shapes, wiring,
# server round-trip), never to interpret resulting success rates.

import argparse
import logging
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from qc.actor import best_of_n_action
from qc.critic import DuelingQChunkCritic, QChunkCritic
from starVLA.model.framework.base_framework import baseframework


class QCPolicyWrapper:
    """Adapter so best_of_n_action can be served through the same
    `.predict_action(**payload)` interface as a plain VLA_JEPA instance."""

    def __init__(self, vla, critic, num_samples, score_kwargs):
        self.vla = vla
        self.critic = critic
        self.num_samples = num_samples
        self.score_kwargs = score_kwargs

    def predict_action(self, **payload):
        return best_of_n_action(
            self.vla,
            self.critic,
            num_samples=self.num_samples,
            **self.score_kwargs,
            **payload,
        )


def main(args) -> None:
    vla = baseframework.from_pretrained(args.ckpt_path)
    device = torch.device(f"cuda:{str(args.cuda)}")
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    action_dim = vla.config.framework.action_model.action_dim
    # NOT the model's native chunk_len (future_action_window_size+1, e.g. 7)
    # -- the critic was trained on chunks truncated to args.horizon_length
    # (qc/train_critic.py's default 5), so its input dim depends on THAT,
    # not the model's own prediction horizon. Mismatching these two raises
    # a state_dict size-mismatch error at load time.
    horizon_length = args.horizon_length
    embed_dim = vla.config.framework.qwenvl.vl_hidden_dim
    proprio_dim = vla.config.framework.action_model.state_dim or 0

    critic_cls = DuelingQChunkCritic if args.critic_type == "dueling" else QChunkCritic
    critic = critic_cls(
        embed_dim=embed_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        horizon_length=horizon_length,
        num_qs=args.num_qs,
    )
    if args.critic_checkpoint_path:
        state_dict = torch.load(args.critic_checkpoint_path, map_location="cpu")
        critic.load_state_dict(state_dict)
    elif not args.allow_untrained_critic:
        raise ValueError(
            "--critic-checkpoint-path is required (pass --allow-untrained-critic "
            "only to smoke-test plumbing -- an untrained critic's scores are meaningless)."
        )
    else:
        logging.warning("Serving with an UNTRAINED critic -- scores/selection are meaningless. Smoke-test only.")
    critic = critic.to(device).eval()

    score_kwargs = dict(
        horizon_length=horizon_length,
        q_agg=args.q_agg,
        uncertainty_penalty=args.uncertainty_penalty,
        actor_disagreement_penalty=args.actor_disagreement_penalty,
        critic_weight=args.critic_weight,
        maximize_score=args.maximize_score,
        selection_mode=args.selection_mode,
        normalize_score_terms=args.normalize_score_terms,
        rank_score_terms=args.rank_score_terms,
        candidate_temperature=args.candidate_temperature,
        candidate_sde_noise_scale=args.candidate_sde_noise_scale,
        candidate_temperature_spread=(
            tuple(args.candidate_temperature_spread) if args.candidate_temperature_spread else None
        ),
        adaptive_entropy_threshold=args.adaptive_entropy_threshold,
        adaptive_resample_temperature=args.adaptive_resample_temperature,
        exploit_temperature=args.exploit_temperature,
        explore_temperature=args.explore_temperature,
        explore_fraction=args.explore_fraction,
        explore_margin=args.explore_margin,
    )
    policy = QCPolicyWrapper(vla, critic, args.num_samples, score_kwargs)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating QC server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"env": "simpler_env", "qc": True},
    )
    logging.info("QC server running (num_samples=%d) ...", args.num_samples)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--critic_checkpoint_path", "--critic-checkpoint-path", dest="critic_checkpoint_path", type=str, default=None)
    parser.add_argument("--allow_untrained_critic", "--allow-untrained-critic", dest="allow_untrained_critic", action="store_true")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--num_samples", "--num-samples", dest="num_samples", type=int, default=8)
    parser.add_argument("--horizon_length", "--horizon-length", dest="horizon_length", type=int, default=5)
    parser.add_argument("--num_qs", "--num-qs", dest="num_qs", type=int, default=5)
    parser.add_argument(
        "--critic_type", "--critic-type", dest="critic_type",
        type=str, default="monolithic", choices=["monolithic", "dueling"],
        help="monolithic = QChunkCritic (qc/train_critic.py checkpoints), dueling = "
        "DuelingQChunkCritic (qc/train_dueling_critic.py checkpoints) -- must match "
        "whichever script trained --critic_checkpoint_path, state_dict keys differ.",
    )
    parser.add_argument("--q_agg", "--q-agg", dest="q_agg", type=str, default="mean", choices=["mean", "min"])
    parser.add_argument("--uncertainty_penalty", "--uncertainty-penalty", dest="uncertainty_penalty", type=float, default=0.0)
    parser.add_argument(
        "--actor_disagreement_penalty", "--actor-disagreement-penalty",
        dest="actor_disagreement_penalty", type=float, default=0.0,
    )
    parser.add_argument("--critic_weight", "--critic-weight", dest="critic_weight", type=float, default=1.0)
    parser.add_argument("--maximize_score", "--maximize-score", dest="maximize_score", action="store_true")
    parser.add_argument(
        "--selection_mode", "--selection-mode", dest="selection_mode",
        type=str, default="score", choices=["score", "majority_vote", "random", "tiered"],
    )
    parser.add_argument(
        "--normalize_score_terms", "--normalize-score-terms", dest="normalize_score_terms",
        action="store_true",
        help="Z-score q/disagreement/actor_disagreement across the N candidates before "
        "combining with the penalty weights, instead of summing raw values. See "
        "qc/actor.py's best_of_n_action docstring for why this matters -- raw terms "
        "live on very different scales and actor_disagreement silently dominates "
        "selection otherwise.",
    )
    parser.add_argument(
        "--rank_score_terms", "--rank-score-terms", dest="rank_score_terms",
        action="store_true",
        help="Rank-transform each score term across the N candidates instead of "
        "z-scoring (see normalize_score_terms) or summing raw. Takes precedence "
        "over normalize_score_terms if both are passed.",
    )
    parser.add_argument(
        "--candidate_temperature", "--candidate-temperature", dest="candidate_temperature",
        type=float, default=1.0,
        help="Scales the flow-matching sampler's INITIAL noise draw (default 1.0 = "
        "original unit-Gaussian behavior). Raising it increases candidate diversity "
        "but risks pushing candidates off-distribution (the model was trained on "
        "unit-Gaussian starts). See qc/actor.py's best_of_n_action for why this "
        "matters -- N candidates were found nearly IDENTICAL in raw action space.",
    )
    parser.add_argument(
        "--candidate_sde_noise_scale", "--candidate-sde-noise-scale", dest="candidate_sde_noise_scale",
        type=float, default=0.0,
        help="Injects noise at every Euler integration step instead of only at the "
        "start (default 0.0 = original pure-ODE behavior, no injection). Likely "
        "gentler on action quality than candidate_temperature for the same amount "
        "of added diversity, since it's spread across the whole trajectory.",
    )
    parser.add_argument(
        "--candidate_temperature_spread", "--candidate-temperature-spread", dest="candidate_temperature_spread",
        type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="Give each of the N candidates its OWN temperature (np.linspace(LOW, HIGH, "
        "num_samples)) instead of one uniform candidate_temperature -- every batch then "
        "spans a range of diversity levels by construction, no threshold to tune, no "
        "second forward pass. Overrides candidate_temperature when set. Can still be "
        "combined with adaptive_entropy_threshold (that would trigger extra resampling "
        "on top if the spread-sampled batch's overall entropy is still too low).",
    )
    parser.add_argument(
        "--adaptive_entropy_threshold", "--adaptive-entropy-threshold", dest="adaptive_entropy_threshold",
        type=float, default=None,
        help="If set, measures the default-temperature candidates' Gaussian-fit "
        "differential entropy (nats) and, if below this threshold, resamples an "
        "EXTRA batch at adaptive_resample_temperature (reusing the already-computed "
        "Qwen encoding, no second expensive forward pass) and merges it into the "
        "candidate pool. Targets the diversity boost at states whose candidates "
        "have actually collapsed, instead of a blanket candidate_temperature bump "
        "which empirically helps some states/categories while hurting others. "
        "Default (unset) disables this entirely, matching prior behavior. Pick a "
        "threshold by checking entropy_estimate values against your own critic/data "
        "-- there's no universal default, it depends on the action space and horizon.",
    )
    parser.add_argument(
        "--adaptive_resample_temperature", "--adaptive-resample-temperature", dest="adaptive_resample_temperature",
        type=float, default=1.3,
        help="Temperature used for the EXTRA resampled batch when "
        "adaptive_entropy_threshold triggers (default 1.3, the setting found best "
        "on one LIBERO-Plus category in today's fixed-temperature sweep).",
    )
    parser.add_argument(
        "--exploit_temperature", "--exploit-temperature", dest="exploit_temperature",
        type=float, default=1.0,
        help="[selection_mode=tiered only] Temperature for the 'exploit' tier -- "
        "candidates kept close to the trained distribution, scored by q-value.",
    )
    parser.add_argument(
        "--explore_temperature", "--explore-temperature", dest="explore_temperature",
        type=float, default=1.6,
        help="[selection_mode=tiered only] Temperature for the 'explore' tier -- "
        "candidates deliberately pushed off-distribution to force a real "
        "disagreement signal to exist, scored by cross-head disagreement.",
    )
    parser.add_argument(
        "--explore_fraction", "--explore-fraction", dest="explore_fraction",
        type=float, default=0.5,
        help="[selection_mode=tiered only] Fraction of num_samples allocated to the "
        "explore tier (rest go to exploit).",
    )
    parser.add_argument(
        "--explore_margin", "--explore-margin", dest="explore_margin",
        type=float, default=1.0,
        help="[selection_mode=tiered only] The explore tier's best disagreement must "
        "exceed this many z-scored units (relative to the exploit tier's OWN "
        "disagreement spread) to be chosen over the exploit pick -- "
        "optimism-under-uncertainty with a margin, not 'always explore.'",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
