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

**Training is staged; inference is a single automated pipeline** (see below). No data or trained
checkpoints are included (see **Data**).

---

# Module 1 — LSEM (Latent Spatial Evidence Module)

A lightweight, interpretable readout of clinical evidence from a **frozen** VLM's **intermediate
token grid**. Two stages:

1. **Latent spatial evidence extraction** (`lsem.extract.LSEMExtractor`). Forward the frozen VLM on
   the RGB + depth images, take the hidden states at a mid layer, and reshape/adaptive-pool the
   image tokens to a fixed grid `H* ∈ R^{M×R×C×D}` (default `2×6×4×D`). All spatial evidence is
   kept intact — nothing is pooled away, routed, or rewritten.

2. **Structure-regularized grid readout** (`lsem.readout.LSEMReadout`). Each concept `c` is read
   with a per-concept **weight field** over the grid, `s_c = Σ_i ⟨e_i, w_{c,i}⟩`. The grid is used
   intact; inductive bias enters only by regularizing the *weight field*:

   ```
   Ω(W_c) = λ_g · Σ_i ‖w_{c,i}‖                         # which cells matter (group)
          + λ_f · Σ_(i,j)∈edges ‖w_{c,i} − w_{c,j}‖      # contiguous regions (fused)
          (+ λ_b · bilateral symmetric/antisymmetric term)
   ```

   With all `λ = 0` this reduces to the plain full-grid linear readout, so the regularized readout
   **cannot underperform it by construction**; the learned field is interpretable (`region_map(c)`
   heatmaps emerge automatically — no hand-defined ROIs, no routing).

---

# Module 2 — CurveToken Grounding

A **training-time** method to stop curve-count collapse in free-generation grounding. Each grounded
curve is preceded by a *carrier token* — by default the object-reference-start token the grounding
template already emits (`<|object_ref_start|>`), so **no new vocabulary is needed**. Its last-layer
hidden state is that curve's latent identity `z_j`. During a light LoRA continue-train the total loss

```
L = L_LM + λ_sep · L_sep ,   L_sep = mean_sample[ mean_{i≠j} relu(cos(z_i,z_j) − m) ]
```

pushes the identities of different curves in the same image apart (use a **negative** margin `m` so
the loss does not saturate at orthogonality). The decoder is unchanged; inference is ordinary
generation. See `curvetoken/` (`separation.py` = the loss, `sft.py` = the SFT integration) and
`curvetoken/example.py`.

**Honest note.** Separating carriers recovers triple-curve cases free generation misses, at a modest
cost to the 2-curve majority — report set-completeness per curve-count `K`, not only the aggregate.

---

# Module 3 — RAEP (Reliability-Aware Evidence Propagation)

An **inference-time** module (no trainable parameters). For each body-surface concept produced by
LSEM it computes a **margin-derived reliability** `r_c ∈ [0,1]` (how decisively the probe lands on
one side), verbalizes the concept with its reliability (High / Moderate / Low), and **abstains** on
a concept whose margin is below threshold. This reliability-annotated evidence is propagated, in a
fixed evidence order, to the diagnosis stage. See `raep/reliability.py`.

**Honest note.** An earlier design also *updated* the reliability with a cross-modal grounding
agreement term (`logit r' = logit r + γ·a`); this update was ablated and did **not** improve over
the static reliability, so the released RAEP uses the static, margin-derived reliability only.

---

# Training (staged) and Inference (single automated pipeline)

**Training** proceeds in three sequential stages (not one joint objective):

- **S1 — Base multimodal SFT.** LoRA-fine-tune the VLM on the multi-turn screening–grounding–
  diagnosis sequences with `L_LM`.
- **S2 — LSEM readout.** With the VLM **frozen**, train the per-concept readout heads on the
  mid-layer (`ℓ*`) token grid: `L_surf = Σ_c CE(y_c, s_c) + λ_g Σ_c Ω_g(W_c) + λ_f Σ_c Ω_f(W_c)`.
- **S3 — CurveToken continue-training.** Resume LoRA fine-tuning with `L_CTG = L_LM + λ_sep·L_sep`.

**Inference** runs as one automated pipeline (`raep.SpineChatPipeline`): given the RGB-D images and
the AP radiograph, the same VLM performs screening (via LSEM + RAEP), ground-first localization, and
structured diagnosis — with reliability-annotated evidence propagated across turns and no manual
intervention.

```python
from raep import SpineChatPipeline
# readout(rgb, depth) -> {"g","d","t","zc"} (LSEM head outputs)
# generate(messages, images, prefix="") -> str (greedy VLM continuation; CurveToken adapter)
pipe = SpineChatPipeline(readout=my_readout, generate=my_generate)
out = pipe.run(rgb_path, depth_path, xray_path)   # -> screening_evidence / grounding / diagnosis
```

---

## Install

```bash
pip install -r requirements.txt
```

`LSEMReadout`, `curvetoken`, and `raep` need only `torch`/`numpy`; `LSEMExtractor` additionally
needs `ms-swift` + `pillow` to run the frozen VLM forward.

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
