# Scoliagent Module A — Reference Implementations

Reference code for the two modules of Scoliagent Module A (fine-tuned VLM for radiation-free
scoliosis screening and diagnosis from RGB-D back-surface images + AP radiographs):

1. **LSEM — Latent Spatial Evidence Module** (`lsem/`): a lightweight, interpretable readout of
   clinical evidence from a **frozen** VLM's **intermediate token grid**, for the screening stage.
2. **CurveToken Grounding** (`curvetoken/`): a training-time method that gives each curve an
   explicit latent identity and pushes the identities apart, so free-generation grounding stops
   collapsing the curve count (notably the rare triple-curve case).

The two are independent — use either on its own. Neither ships data (see **Data**).

---

# Module 1 — LSEM (Latent Spatial Evidence Module)

LSEM is a lightweight, interpretable readout of clinical evidence from a **frozen** vision–language
model's **intermediate token grid**, for radiation-free scoliosis screening from RGB-D
back-surface images. Two stages:

1. **Latent spatial evidence extraction** (`lsem.extract.LSEMExtractor`).
   Forward the frozen VLM on the RGB + depth back-surface images, take the hidden states at a
   mid layer, and reshape/adaptive-pool the RGB and depth image tokens to a fixed spatial grid —
   a structure-preserving tensor `H* ∈ R^{M×R×C×D}` (default `2×6×4×D`). All spatial evidence is
   kept; nothing is pooled away, routed, or rewritten.

2. **Structure-regularized grid readout** (`lsem.readout.LSEMReadout`).
   Each clinical concept `c` is read with a per-concept **weight field** over the grid,

   ```
   s_c = Σ_i ⟨e_i, w_{c,i}⟩          e_i = P h_i   (one weight vector per grid cell i)
   ```

   The grid `H*` is used **intact**. Inductive bias is imposed only by regularizing the *weight
   field*:

   ```
   Ω(W_c) = λ_group     · Σ_i ‖w_{c,i}‖                              # which cells matter
          + λ_fused     · Σ_(i,j)∈edges ‖w_{c,i} − w_{c,j}‖          # contiguous regions
          + λ_bilateral · Σ_(i,i')∈mirror (‖w_{c,i}+w_{c,i'}‖ + ‖w_{c,i}−w_{c,i'}‖)   # symmetric / antisymmetric
   ```

   With all `λ = 0` this is exactly the plain full-grid linear readout, so the regularized
   readout **cannot underperform it by construction**. The learned weight field is interpretable:
   `region_map(c)` gives a per-cell importance heatmap (readout regions emerge automatically — no
   hand-defined ROIs and no routing), and `bilateral_energy(c)` reports how much a concept relies
   on left–right asymmetry.

**Design principle — preserve first, regularize the *reading* — not the features.** The evidence
grid is never compressed or altered; structure enters only through the readout weight field. This
avoids the information bottleneck / representation-collapse failure modes of pooling, attention,
slot, and relational readouts.

### Usage

See `lsem`'s example. Sketch:

```python
from lsem import LSEMExtractor, LSEMReadout, load_swift_vlm
model, template = load_swift_vlm("Qwen/Qwen3.5-4B", adapter="<your_adapter_dir>")
extractor = LSEMExtractor(model, template, layer=16, grid=(6, 4))
grid = extractor.extract(rgb_path, depth_path, SYSTEM, USER)      # (2, 6, 4, D)
readout = LSEMReadout(in_dim=grid.shape[-1], num_classes={"lateral": 2, "shoulder": 3, "trunk": 3})
logits = readout(grid.unsqueeze(0), "lateral")
# train: loss = Σ_c CE(readout(grid, c), y_c) + readout.structure_penalty(c)
```

`LSEMReadout` needs only `torch`/`numpy`; `LSEMExtractor` additionally needs `ms-swift` + `pillow`
to run the frozen VLM forward. The mid layer (`layer=16`) and grid `(6,4)` are the reference
settings; LSEM is a *lightweight, interpretable, floor-guaranteed* module — the contribution is
the design and the emergent interpretability, not a large accuracy gain over the plain readout.

---

# Module 2 — CurveToken Grounding

A **training-time** method to stop curve-count collapse in free-generation grounding. When a
scoliosis case has multiple curves (1 major + up to 2 compensatory), the model tends to collapse
the count — most often dropping the third, faintest compensatory curve of a **triple-curve** case.
CurveToken Grounding fixes this on the representation side while leaving the decoder untouched:

1. **Per-curve carrier token.** Every grounded curve is preceded by a *carrier token* — by default
   the object-reference-start token the grounding template already emits (`<|object_ref_start|>`),
   so **no vocabulary change / embedding resize is needed**. The last-layer hidden state at that
   position is taken as that curve's latent identity `z_j`.

2. **Instance-separation loss** (`curvetoken.separation.CurveTokenSeparation`). During a light
   LoRA continue-train, push the identities of *different* curves in the *same* image apart:

   ```
   L = L_LM  +  sep_weight · L_sep
   L_sep = mean_sample [ mean_{i≠j} relu( cos(z_i, z_j) − margin ) ]
   ```

   `L_LM` is the standard grounding SFT loss (keeps boxes/findings intact). Use a **negative**
   margin (default `-0.3`): with `margin = 0` the loss saturates the moment the carriers are merely
   orthogonal (trivial in high dimension) and then supplies no gradient, so it never reshapes the
   encoder; a negative margin keeps the separation signal active throughout training.

Inference is ordinary free generation — no decoder change, no count head, no extra pass. The
separation gradient flows into the (LoRA-adapted) visual encoder, sharpening the weakly-encoded
third curve so the model stops merging it away.

### Usage

See `curvetoken/example.py` (runs a standalone loss demo with only `torch`). Sketch of the SFT
integration:

```python
from transformers import Trainer
from curvetoken import CurveTokenSFTMixin, build_separation

class CurveTokenTrainer(CurveTokenSFTMixin, Trainer):
    pass

trainer = CurveTokenTrainer(model=lora_model, args=..., train_dataset=grounding_ds, ...)
trainer.curvetoken_sep = build_separation(tokenizer)   # <|object_ref_start|> carrier, margin -0.3
trainer.curvetoken_sep_weight = 0.3                    # L = L_LM + 0.3 * L_sep
trainer.train()
```

Reference settings: carrier `<|object_ref_start|>`, `margin = -0.3`, `sep_weight = 0.3`, a light
all-linear LoRA continue-train (~2 epochs) of an already-grounding-trained VLM. During training
watch `stats["mean_cos"]` fall toward the margin (confirming the loss stays active).

`curvetoken` needs only `torch` for the loss; the SFT mixin plugs into a HuggingFace / ms-swift
`Trainer` and forces `output_hidden_states=True` + `use_cache=False` (the latter is required for
hybrid-attention backbones such as Qwen3.5 during training).

**Honest note on the tradeoff.** Separating carriers recovers triple-curve cases that free
generation misses entirely, and improves overall set-completeness, but pushing separation hard can
**regress the majority (2-curve) case**. Report completeness per curve-count `K`, not only the
aggregate, and tune `sep_weight` / `margin` against a held-out split — not the test set.

---

## Install

```bash
pip install -r requirements.txt
```

## Data

**No data is included** (back-surface photographs and radiographs are protected patient data).
Provide your own dataset: LSEM's extractor takes paths to a subject's RGB image and depth map with
per-concept clinical labels; CurveToken Grounding takes a standard grounding SFT dataset (curve
reference + box per curve). No landmark / spine-centerline masks are used by either module.

## Backbones

The reference code targets a Qwen-family VLM via ms-swift (single image-placeholder id +
`image_grid_thw`; grounding tokens `<|object_ref_start|>` / `<|box_start|>`). For other backbones,
pass your own `model`/`template`, set `img_token_id` / `merge` for LSEM, and pass the appropriate
carrier-token string to `build_separation` for CurveToken.

## Citation

```bibtex
@article{scoliagent_moduleA,
  title  = {Latent Spatial Evidence and CurveToken Grounding for Radiation-Free Scoliosis Screening and Diagnosis},
  author = {...},
  year   = {2026}
}
```

## License

TBD (add before public release).
