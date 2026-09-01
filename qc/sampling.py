"""Best-of-N candidate action sampling for VLA-JEPA's flow-matching head.

FlowmatchingActionHead.predict_action (starVLA/model/modules/action_model/
GR00T_ActionHeader.py) is already batch-size-general: `batch_size =
vl_embs.shape[0]`, and it draws that many INDEPENDENT noise trajectories and
denoises them all in one call (AdaLN throughout, no batch-norm-style
batch-dependent state, so this is safe regardless of N). So getting N
diverse candidates for a single observation only requires running the
expensive Qwen3-VL forward pass ONCE and then repeating its output
(embodied_action_tokens) N times along the batch dim before handing it to
their (UNMODIFIED) action head -- no changes to any of their model code.

This mirrors VLA_JEPA.predict_action step-for-step up through computing
embodied_action_tokens, then diverges only in the repeat-and-batch-decode
step.
"""

import numpy as np
import torch

from starVLA.training.trainer_utils.trainer_tools import resize_images


@torch.inference_mode()
def predict_action_candidates(model, batch_images, instructions, state=None, num_samples=8, **kwargs):
    """Same call signature/shape conventions as VLA_JEPA.predict_action, but
    returns num_samples candidate action chunks for the SINGLE observation
    in batch_images[0]/instructions[0].

    Returns:
        dict with:
          normalized_actions: [num_samples, chunk_len, action_dim] np.ndarray
          embodied_action_tokens: [1, num_tokens, H] np.ndarray -- the ONE
            Qwen forward's output (identical for every candidate, so NOT
            repeated here; the caller pools/repeats it for the critic).
    """
    assert len(batch_images) == 1 and len(instructions) == 1, (
        "predict_action_candidates samples N candidates for a SINGLE live observation "
        "(that's what best-of-N eval calls one env step at a time with); "
        "a batch of >1 observations should use the plain predict_action instead."
    )

    train_obs_image_size = getattr(model.config.datasets.vla_data, "image_size", None)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images,
        instructions=instructions,
        prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt},
    )

    embodied_action_indices = torch.isin(
        qwen_inputs["input_ids"],
        torch.tensor([model.embodied_action_token_id], device=qwen_inputs["input_ids"].device),
    )
    embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = qwenvl_outputs.hidden_states[-1]
        B, _, H = last_hidden.shape
        embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

    state_t = (
        torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
        if state is not None
        else None
    )

    # The single expensive forward pass is done -- everything below is cheap
    # repeat + the (already batch-general) flow-matching decode.
    tokens_rep = embodied_action_tokens.repeat(num_samples, 1, 1)
    state_rep = state_t.repeat(num_samples, 1, 1) if state_t is not None else None

    with torch.autocast("cuda", dtype=torch.float32):
        pred_actions = model.action_model.predict_action(tokens_rep, state_rep)  # [N, chunk_len, action_dim]

    return {
        "normalized_actions": pred_actions.detach().cpu().numpy(),
        "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy(),
    }
