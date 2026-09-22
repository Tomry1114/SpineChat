"""Evidence-ordered inference pipeline: screen -> localize (ground-first) -> diagnose.

This ties the three SpineChat components into a single automated inference pass:
  1. LSEM readout produces body-surface screening concepts from the RGB-D images;
  2. RAEP serializes them into reliability-annotated evidence (``raep.format_evidence``);
  3. the VLM localizes the curves on the AP radiograph **ground-first** (without seeing the
     screening evidence, to avoid contaminating localization), then decodes the structured
     diagnosis conditioned on the grounding boxes AND the propagated evidence.

Training is staged; **inference is a single automated pipeline** (no manual step between turns).

Model-specific decoding is injected as a callable so this module stays framework-light:
    generate(messages, images, prefix="") -> str      # greedy VLM continuation
    readout(rgb_path, depth_path)        -> dict       # LSEM head outputs {g,d,t,zc}
"""
from __future__ import annotations
from .reliability import format_evidence

SYS = ("You are a scoliosis screening-and-diagnosis assistant. First screen from the RGB-D "
       "body-surface images (radiation-free); if further work-up is advised, localize every curve "
       "(major and compensatory) on the AP radiograph and give the structured findings.")
U1 = ("<image><image>\nThese are the back-surface images of the same subject (image 1: RGB "
      "photograph, image 2: depth map). Perform radiation-free scoliosis screening: report the "
      "shoulder balance, the trunk shift, and the dominant lateral tendency, then state whether an "
      "X-ray is advised.")
U2 = ("<image>\nThis is the subject's AP spine radiograph. Localize each scoliotic curve (labelling "
      "the major and compensatory curves) and give the structured diagnosis.")
NEUTRAL = ("Body-surface screening completed; an AP radiograph was obtained for further assessment.")
DIAG_FIELD = "Appearance abnormality:"   # first structured-diagnosis field; grounding = text before it


def _diag_messages(turn1: str):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": U1},
            {"role": "assistant", "content": turn1},
            {"role": "user", "content": U2}]


class SpineChatPipeline:
    """Evidence-ordered inference orchestrator.

    Args:
        readout:  callable(rgb_path, depth_path) -> {"g","d","t","zc"} (LSEM head outputs).
        generate: callable(messages, images, prefix="") -> str (VLM greedy continuation; the
                  adapter should be the CurveToken-trained one so grounding handles K=3).
    """

    def __init__(self, readout, generate):
        self.readout = readout
        self.generate = generate

    def run(self, rgb_path: str, depth_path: str, xray_path: str) -> dict:
        imgs = [rgb_path, depth_path, xray_path]
        # 1-2. screen + RAEP reliability-annotated evidence
        evidence = format_evidence(self.readout(rgb_path, depth_path))
        # 3a. ground-first: localize WITHOUT the screening evidence (neutral Turn-1); keep ONLY the
        #     curve-localization part (everything before the first diagnosis field) as the prefix.
        grounding = self.generate(_diag_messages(NEUTRAL), imgs, prefix="").split(DIAG_FIELD)[0]
        # 3b. diagnosis: continue from the grounding prefix, now conditioned on the propagated evidence.
        diagnosis = grounding + self.generate(_diag_messages(evidence), imgs, prefix=grounding)
        return {"screening_evidence": evidence, "grounding": grounding, "diagnosis": diagnosis}
