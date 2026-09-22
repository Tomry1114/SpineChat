"""CurveToken Grounding — per-curve carrier identity + instance-separation loss.

A training-time method to stop curve-count collapse in free-generation grounding (in particular
the rare triple-curve case). Each curve's carrier token (by default the object-reference-start
token the grounding template already emits — no new vocabulary) gives that curve an explicit
latent identity; an auxiliary instance-separation loss keeps the identities of different curves
in the same image apart. The decoder is unchanged and inference is ordinary free generation.

    L = L_LM + sep_weight * L_sep

See ``separation.CurveTokenSeparation`` (the loss) and ``sft`` (how to add it to a grounding SFT).
"""
from .separation import CurveTokenSeparation
from .sft import (CurveTokenSFTMixin, build_separation, curvetoken_total_loss,
                  DEFAULT_CARRIER_TOKEN, DEFAULT_MARGIN, DEFAULT_SEP_WEIGHT)

__all__ = ["CurveTokenSeparation", "CurveTokenSFTMixin", "build_separation",
           "curvetoken_total_loss", "DEFAULT_CARRIER_TOKEN", "DEFAULT_MARGIN", "DEFAULT_SEP_WEIGHT"]
__version__ = "0.1.0"
