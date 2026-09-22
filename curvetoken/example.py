"""Minimal CurveToken Grounding usage.

(1) Standalone loss on a synthetic batch — runnable with only torch, no model/data:
      python -m curvetoken.example
(2) Sketch of plugging it into a grounding SFT (commented; needs your VLM + Trainer).
"""
import torch
from curvetoken import CurveTokenSeparation


def demo_loss():
    torch.manual_seed(0)
    H, T = 32, 40
    CARRIER = 7                                   # pretend this id is <|object_ref_start|>
    input_ids = torch.randint(0, 100, (2, T))
    input_ids[0, [5, 18, 30]] = CARRIER           # sample 0: three curves
    input_ids[1, [9]] = CARRIER                   # sample 1: one curve (no pair -> no loss)
    hidden = torch.randn(2, T, H, requires_grad=True)

    sep = CurveTokenSeparation(CARRIER, margin=-0.3)
    loss, stats = sep(hidden, input_ids)
    loss.backward()
    print("loss=%.4f  stats=%s  grad_ok=%s" % (
        float(loss), stats, hidden.grad is not None and torch.isfinite(hidden.grad).all().item()))


# --- (2) Grounding SFT integration sketch -------------------------------------------------
# from transformers import Trainer
# from curvetoken import CurveTokenSFTMixin, build_separation
#
# class CurveTokenTrainer(CurveTokenSFTMixin, Trainer):
#     pass
#
# trainer = CurveTokenTrainer(model=lora_model, args=..., train_dataset=grounding_ds, ...)
# trainer.curvetoken_sep = build_separation(tokenizer)   # <|object_ref_start|> carrier, margin -0.3
# trainer.curvetoken_sep_weight = 0.3
# trainer.train()
# Watch stats["mean_cos"] fall toward the margin; keep an eye on curve recall for the majority
# (2-curve) case, which can regress if separation is pushed too hard.

if __name__ == "__main__":
    demo_loss()
