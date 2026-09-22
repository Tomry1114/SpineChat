"""CurveToken instance-separation loss.

Each grounded curve in the model's output is preceded by a *carrier token* — by default the
object-reference-start token the grounding template already emits (``<|object_ref_start|>``),
so **no vocabulary change is needed**. The last-layer hidden state at that position is taken as
that curve's latent identity ``z_j``.

The separation loss pushes the identities of *different* curves in the *same* sample apart, so
the decoder maintains one distinct representation per curve and stops collapsing the curve count
(notably the rare triple-curve case, which free generation otherwise merges into two):

    L_sep = mean_sample [ mean_{i != j} relu( cos(z_i, z_j) - margin ) ]

Use a **negative** margin. With ``margin = 0`` the loss saturates the instant the carriers become
merely orthogonal (trivial in a high-dimensional space) and then supplies no gradient; a negative
margin keeps pushing the carriers toward anti-correlation, so the separation signal stays active
throughout training.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveTokenSeparation(nn.Module):
    """Instance-separation loss over per-curve carrier hidden states.

    Args:
        carrier_token_id: vocab id of the per-curve carrier token (e.g.
            ``tokenizer.convert_tokens_to_ids("<|object_ref_start|>")``).
        margin: cosine hinge threshold; the loss is active while ``cos > margin``. Negative
            values (default ``-0.3``) keep the carriers separating past orthogonality.
    """

    def __init__(self, carrier_token_id: int, margin: float = -0.3):
        super().__init__()
        self.carrier_token_id = int(carrier_token_id)
        self.margin = float(margin)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """Compute the batch-mean separation loss.

        Args:
            hidden_states: ``(B, T, H)`` last-layer hidden states (must carry grad).
            input_ids:     ``(B, T)`` token ids aligned with ``hidden_states``.

        Returns:
            ``(loss, stats)`` where ``loss`` is a scalar tensor (0 if no sample has >=2 curves)
            and ``stats`` is a dict with ``n_samples_with_curves``, ``mean_cos`` (raw mean
            off-diagonal cosine — watch it fall toward ``margin``), and ``mean_k``.
        """
        if hidden_states.dim() == 2:  # allow a single unbatched sample (T, H)
            hidden_states = hidden_states.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
        total = hidden_states.new_zeros(())
        n_used, cos_sum, k_sum = 0, 0.0, 0
        for b in range(hidden_states.shape[0]):
            pos = (input_ids[b] == self.carrier_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() < 2:  # need >=2 curves in the sample to separate anything
                continue
            z = F.normalize(hidden_states[b, pos].float(), dim=-1)          # (k, H)
            k = z.shape[0]
            sim = z @ z.t()
            off = sim[~torch.eye(k, dtype=torch.bool, device=z.device)]     # off-diagonal cosines
            total = total + F.relu(off - self.margin).mean()
            cos_sum += float(off.mean()); k_sum += k; n_used += 1
        if n_used == 0:
            return hidden_states.new_zeros(()), {"n_samples_with_curves": 0, "mean_cos": 0.0, "mean_k": 0.0}
        return total / n_used, {"n_samples_with_curves": n_used,
                                "mean_cos": cos_sum / n_used, "mean_k": k_sum / n_used}
