#!/usr/bin/env python
"""SpineChat Stage 3 — CurveToken continue-training of the base LoRA VLM.

Resumes the Stage-1 LoRA adapter and continues LoRA fine-tuning with  L = L_LM + λ_sep · L_sep,
where L_sep separates the per-curve carrier tokens (see ``curvetoken``). Decoder unchanged;
inference is ordinary generation. Requires ms-swift + a Qwen-family VLM and a grounding SFT jsonl
(messages / images / objects: {ref, bbox} per curve). No data ships with this repo.
"""
import argparse, json, math
import numpy as np
import torch
from swift.model.register import get_model_processor
from swift.template import get_template
from swift import Swift
from curvetoken import build_separation, curvetoken_total_loss, DEFAULT_MARGIN, DEFAULT_SEP_WEIGHT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", required=True, help="Stage-1 LoRA checkpoint to continue")
    ap.add_argument("--data", required=True, help="grounding SFT jsonl")
    ap.add_argument("--out", default="outputs/s3_curvetoken")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--sep_weight", type=float, default=DEFAULT_SEP_WEIGHT)   # 0.3
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)            # -0.3
    ap.add_argument("--max_pixels", type=int, default=1003520)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = get_model_processor(args.model, load_model=True)
    model = Swift.from_pretrained(model, args.adapter, is_trainable=True)      # continue Stage-1 LoRA
    model.config.use_cache = False; model.to(dev); model.train()
    tok = getattr(proc, "tokenizer", proc)
    sep = build_separation(tok, margin=args.margin)                           # carrier = <|object_ref_start|>
    tmpl = get_template(proc, max_pixels=args.max_pixels, max_length=4096); tmpl.set_mode("train")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    rows = [json.loads(l) for l in open(args.data)]
    step = 0; opt.zero_grad()
    for ep in range(math.ceil(args.epochs)):
        for it, ri in enumerate(np.random.permutation(len(rows))):
            r = rows[int(ri)]
            enc = tmpl.encode({"messages": r["messages"], "images": r["images"], "objects": r.get("objects")})
            b = tmpl.data_collator([enc])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(**b, output_hidden_states=True, use_cache=False)      # hybrid-attn: cache off
            loss, stats = curvetoken_total_loss(out.loss, out.hidden_states[-1], b["input_ids"], sep, args.sep_weight)
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"ep{ep} step{step} lm={stats['lm']:.3f} sep={stats['sep']:.3f} mean_cos={stats['mean_cos']:+.3f}", flush=True)
    model.save_pretrained(args.out); print("saved", args.out)


if __name__ == "__main__":
    main()
