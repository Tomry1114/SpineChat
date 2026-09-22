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

`LSEMReadout`, `curvetoken`, and `raep` need only `torch`/`numpy`; `LSEMExtractor` additionally
needs `ms-swift` + `pillow` to run the frozen VLM forward. See `train/README.md` for the training
recipe and `raep.SpineChatPipeline` for the inference entry point.

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

## Citation

```bibtex
@article{spinechat,
  title  = {SpineChat: Latent Spatial Evidence, CurveToken Grounding, and Reliability-Aware
            Evidence Propagation for Radiation-Free Scoliosis Screening and Diagnosis},
  author = {...},
  year   = {2026}
}
```

## License

TBD (add before public release).
