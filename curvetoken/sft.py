"""Integrating CurveToken separation into a grounding SFT step.

CurveToken Grounding is a *training-time* method: the decoder is unchanged, and inference is
ordinary free generation. During a light LoRA continue-train of a grounding VLM the total loss is

    L = L_LM  +  sep_weight * L_sep

where ``L_LM`` is the standard grounding SFT loss (keeps boxes/findings intact) and ``L_sep`` is
the :class:`~curvetoken.separation.CurveTokenSeparation` loss on the per-curve carrier hiddens.
"""
from __future__ import annotations
import torch
from .separation import CurveTokenSeparation

# Empirically effective settings (Qwen3.5-VL grounding; light all-linear LoRA continue-train).
DEFAULT_CARRIER_TOKEN = "<|object_ref_start|>"   # one per curve; reuses existing vocab (no resize)
DEFAULT_MARGIN = -0.3
DEFAULT_SEP_WEIGHT = 0.3


def build_separation(tokenizer, carrier_token: str = DEFAULT_CARRIER_TOKEN,
                     margin: float = DEFAULT_MARGIN) -> CurveTokenSeparation:
    """Resolve the carrier token to an id and build the loss module."""
    cid = tokenizer.convert_tokens_to_ids(carrier_token)
    if cid is None or cid < 0 or cid == getattr(tokenizer, "unk_token_id", -1):
        raise ValueError(f"carrier token {carrier_token!r} maps to an invalid id {cid}")
    return CurveTokenSeparation(cid, margin=margin)


def curvetoken_total_loss(lm_loss: torch.Tensor, hidden_states: torch.Tensor,
                          input_ids: torch.Tensor, sep: CurveTokenSeparation,
                          sep_weight: float = DEFAULT_SEP_WEIGHT):
    """``L_LM + sep_weight * L_sep``. Returns ``(total_loss, stats)``."""
    l_sep, stats = sep(hidden_states, input_ids)
    total = lm_loss + sep_weight * l_sep
    return total, {"lm": float(lm_loss.detach()), "sep": float(l_sep.detach()), **stats}


class CurveTokenSFTMixin:
    """Trainer mixin (HuggingFace / ms-swift) adding CurveToken separation to ``compute_loss``.

    Subclass your ``Trainer`` with this mixin first, then set on the instance::

        trainer.curvetoken_sep = build_separation(tokenizer)   # or None to disable
        trainer.curvetoken_sep_weight = 0.3

    The mixin forces ``output_hidden_states=True`` / ``use_cache=False`` and reads the carrier
    hiddens from the last layer. (For hybrid-attention backbones such as Qwen3.5, ``use_cache``
    must be off during training to avoid an incompatible cache path.)
    """

    curvetoken_sep: CurveTokenSeparation | None = None
    curvetoken_sep_weight: float = DEFAULT_SEP_WEIGHT

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        loss = outputs.loss
        if self.curvetoken_sep is not None:
            loss, _ = curvetoken_total_loss(
                outputs.loss, outputs.hidden_states[-1], inputs["input_ids"],
                self.curvetoken_sep, self.curvetoken_sep_weight)
        return (loss, outputs) if return_outputs else loss
