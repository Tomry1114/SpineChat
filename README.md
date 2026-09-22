# SpineChat

Reference code for **SpineChat**, a vision–language framework for radiation-free scoliosis
screening and diagnosis from RGB-D back-surface images + AP radiographs. SpineChat has three
components:

1. **LSEM — Latent Spatial Evidence Module** (`lsem/`): a lightweight, interpretable readout of
   clinical evidence from a **frozen** VLM's **intermediate token grid**, for screening.
2. **CurveToken Grounding** (`curvetoken/`): a training-time method giving each curve an explicit
   latent identity + an instance-separation loss, so free-generation grounding stops collapsing
   the curve count (notably the rare triple-curve case).
3. **RAEP — Reliability-Aware Evidence Propagation** (`raep/`): an **inference-time** module that
   attaches a margin-derived reliability to each screening concept and propagates the
   reliability-annotated evidence, in a fixed evidence order, to the radiographic diagnosis.

## Install

```bash
pip install -r requirements.txt
```

## Training and Inference

**Training**

```bash
# S1  base multimodal SFT (LoRA, language-modeling loss)
MODEL=Qwen/Qwen3.5-4B DATA=data/train.jsonl bash train/s1_base_sft.sh

# S2  LSEM readout on the frozen VLM's mid-layer grid
#     (feats_train.npz is produced by lsem.extract.LSEMExtractor: X + y_<concept>)
python train/s2_lsem.py --features feats_train.npz \
    --concepts lateral shoulder trunk xray --out lsem_readout.pt

# S3  CurveToken continue-training (resumes the S1 adapter)
python train/s3_curvetoken.py --model Qwen/Qwen3.5-4B \
    --adapter outputs/s1_base_sft/checkpoint-XXX \
    --data data/grounding_train.jsonl --out outputs/s3_curvetoken
```

**Inference** — end-to-end (screen → ground-first localize → structured diagnosis):

```bash
python inference.py --model Qwen/Qwen3.5-4B --adapter outputs/s3_curvetoken \
    --lsem lsem_readout.pt --rgb rgb.png --depth depth.png --xray xray.png
```

## Data

**No data is included** (back-surface photographs and radiographs are protected patient data).
Provide your own dataset: LSEM takes a subject's RGB image + depth map with per-concept clinical
labels; CurveToken takes a standard grounding SFT dataset (curve reference + box per curve). No
landmark / spine-centerline masks are used by any module.

## Backbones

The reference code targets a Qwen-family VLM via ms-swift (single image-placeholder id +
`image_grid_thw`; grounding tokens `<|object_ref_start|>` / `<|box_start|>`). For other backbones,
pass your own `model`/`template`, set `img_token_id` / `merge` for LSEM, and pass the appropriate
carrier-token string to CurveToken.

## License

TBD (add before public release).
