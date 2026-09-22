# Training (three sequential stages)

SpineChat is trained in **three sequential stages** (not one joint objective). No data ships with
this repo — provide your own (see the top-level **Data** section).

### Stage 1 — Base multimodal SFT
LoRA-fine-tune the VLM on the multi-turn screening→grounding→diagnosis sequences with the
language-modeling loss `L_LM`. Produces the base adapter used by Stages 2 and 3.
```bash
MODEL=Qwen/Qwen3.5-4B DATA=data/train.jsonl bash train/s1_base_sft.sh
```

### Stage 2 — LSEM readout (VLM frozen)
With the VLM **frozen**, train the per-concept structure-regularized readout heads on the
mid-layer (`ℓ*=16`) token grid: `L = Σ_c CE(y_c, s_c) + Ω_g(W_c) + Ω_f(W_c)`. First extract the
latent grids with `lsem.extract.LSEMExtractor` and save them as an `.npz` (`X`, `y_<concept>`).
```bash
python train/s2_lsem.py --features feats_train.npz \
    --concepts lateral shoulder trunk xray --out lsem_readout.pt
```

### Stage 3 — CurveToken continue-training
Resume the Stage-1 LoRA adapter and continue with `L = L_LM + λ_sep·L_sep` (λ_sep=0.3, margin=−0.3).
```bash
python train/s3_curvetoken.py --model Qwen/Qwen3.5-4B \
    --adapter outputs/s1_base_sft/checkpoint-XXX \
    --data data/grounding_train.jsonl --out outputs/s3_curvetoken
```

**Inference** then runs as a single automated pipeline — see `raep.SpineChatPipeline` in the
top-level README.

### Reference hyperparameters
| stage | LoRA | lr | epochs | notes |
|---|---|---|---|---|
| S1 base | r32/α64, all-linear, freeze_vit=false (vit_lr 2e-5) | 1e-4 | 3 | bs1×accum16, cosine, wd 0.1 |
| S2 LSEM | — (readout head only) | 2e-3 | 300 | Adam wd 1e-4, ℓ*=16, proj_dim 96, λ_g=λ_f=0.01 |
| S3 CurveToken | continue S1 (all-linear) | 5e-5 | 2 | accum8, λ_sep=0.3, margin −0.3 |
