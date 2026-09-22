#!/usr/bin/env python
"""SpineChat end-to-end inference: screen -> ground-first localize -> structured diagnosis.

Loads a CurveToken-trained Qwen-family VLM (ms-swift) + the Stage-2 LSEM readout heads, then runs
``raep.SpineChatPipeline`` on one subject. Runnable once you provide a trained adapter, trained LSEM
readout weights, and the three images.

    python inference.py --model Qwen/Qwen3.5-4B --adapter outputs/s3_curvetoken \
        --lsem lsem_readout.pt --rgb rgb.png --depth depth.png --xray xray.png
"""
import argparse
import torch
from swift.model.register import get_model_processor
from swift.template import get_template
from swift import Swift
from lsem import LSEMExtractor, LSEMReadout
from raep import SpineChatPipeline
from raep.pipeline import SYS, U1

# --- label-index conventions of the Stage-2 readout (LSEMReadout uses sorted class labels) ---
# Confirm these against your own label ordering; they only affect the sign/side of the evidence.
RIGHT_IDX = 1     # index of the "right" class for the binary `lateral` (curve side) head
ADVISE_IDX = 1    # index of the "x-ray advised" class for the binary `xray` head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="CurveToken-trained LoRA adapter (Stage 3)")
    ap.add_argument("--lsem", required=True, help="Stage-2 lsem_readout.pt")
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--xray", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=1003520)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, proc = get_model_processor(args.model, load_model=True)
    model = Swift.from_pretrained(model, args.adapter); model.eval(); model.to(dev)
    model.config.use_cache = True
    tok = getattr(proc, "tokenizer", proc)
    tmpl = get_template(proc, max_pixels=args.max_pixels, max_length=4096); tmpl.set_mode("train")

    # --- generate(messages, images, prefix) : greedy VLM continuation with a closed-think primer ---
    NT = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)   # force structured decode
    def generate(messages, images, prefix=""):
        rec = {"messages": messages + [{"role": "assistant", "content": "ok"}], "images": images}
        b = tmpl.data_collator([tmpl.encode(rec)])
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        lab = b["labels"][0].cpu().numpy()
        a0 = int((lab != -100).nonzero()[0][0])                            # assistant span start
        ids, am = b["input_ids"][:, :a0], b["attention_mask"][:, :a0]
        inj = torch.tensor([NT + (tok.encode(prefix, add_special_tokens=False) if prefix else [])], device=dev)
        ids = torch.cat([ids, inj], 1); am = torch.cat([am, torch.ones_like(inj)], 1)
        g = dict(input_ids=ids, attention_mask=am)
        for k in ("pixel_values", "image_grid_thw"):
            if k in b: g[k] = b[k]
        with torch.no_grad():
            out = model.generate(**g, max_new_tokens=args.max_new_tokens, do_sample=False)
        return proc.decode(out[0][ids.shape[1]:], skip_special_tokens=False)

    # --- readout(rgb, depth) -> {g,d,t,zc} via LSEM (frozen VLM grid + trained heads) ---
    ext = LSEMExtractor(model, tmpl, layer=args.layer, device=dev)
    ck = torch.load(args.lsem, map_location="cpu")
    in_dim = ck["state_dict"]["proj.weight"].shape[1]
    head = LSEMReadout(in_dim=in_dim, num_classes=ck["num_classes"], grid_shape=ck["grid_shape"])
    head.load_state_dict(ck["state_dict"]); head.eval()

    def readout(rgb, depth):
        grid = ext.extract(rgb, depth, SYS, U1).unsqueeze(0)               # 1, M, R, C, D
        with torch.no_grad():
            logit = {c: head(grid, c)[0] for c in ck["num_classes"]}
        prob = {c: torch.softmax(v, 0) for c, v in logit.items()}
        return {  # g,d = probabilities; t,zc = signed logit margins (calibrate RAEP thresholds to this scale)
            "g":  float(prob["xray"][ADVISE_IDX]),
            "d":  float(prob["lateral"][RIGHT_IDX]),
            "t":  float(logit["trunk"][RIGHT_IDX] - logit["trunk"][1 - RIGHT_IDX]),
            "zc": float(logit["shoulder"][1 - RIGHT_IDX] - logit["shoulder"][RIGHT_IDX]),
        }

    out = SpineChatPipeline(readout=readout, generate=generate).run(args.rgb, args.depth, args.xray)
    print("=== SCREENING EVIDENCE ===\n" + out["screening_evidence"])
    print("\n=== GROUNDING ===\n" + out["grounding"])
    print("\n=== DIAGNOSIS ===\n" + out["diagnosis"])


if __name__ == "__main__":
    main()
