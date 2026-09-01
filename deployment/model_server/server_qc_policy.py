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
from qc.critic import QChunkCritic
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

    critic = QChunkCritic(
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
        type=str, default="score", choices=["score", "majority_vote"],
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
