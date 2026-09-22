"""RAEP — Reliability-Aware Evidence Propagation (reliability quantification + verbalization).

RAEP is an **inference-time** module with **no trainable parameters**. It (i) attaches a
margin-derived reliability r_c in [0,1] to each body-surface screening concept produced by the
LSEM readout, and (ii) serializes each concept together with its reliability as natural-language
evidence, which is then propagated — in a fixed evidence order (screen -> localize -> diagnose,
see ``pipeline.py``) — to the downstream radiographic diagnosis.

Reliability is the readout's **margin** mapped to [0,1] (how decisively the probe lands on one
side); a concept whose margin is below an abstention threshold is dropped rather than asserted.

Note: an earlier design additionally *updated* the reliability with a cross-modal grounding
agreement term (logit r' = logit r + gamma * a). That update was ablated and did not improve over
the static reliability, so the released RAEP uses the **static, margin-derived** reliability only.
"""
from __future__ import annotations

# thresholds (match the reference implementation)
TAU_TRUNK_SIGN = 0.01     # |t| below this -> "no clear trunk asymmetry"
TRUNK_SCALE = 0.03        # trunk reliability saturates at |t| = TRUNK_SCALE
SHOULDER_SCALE = 2.5      # shoulder reliability saturates at |z| = SHOULDER_SCALE
SHOULDER_ABSTAIN = 1.0    # |z| below this -> abstain on shoulder laterality


def confidence_word(r: float) -> str:
    """Map a reliability r in [0,1] to a coarse confidence word."""
    return "High" if r > 0.5 else "Moderate" if r > 0.25 else "Low"


def concept_reliability(readout: dict) -> dict:
    """Margin-derived reliability r_c in [0,1] per screening concept.

    ``readout`` holds the LSEM head outputs:
        g  : P(x-ray advised)         in [0,1]
        d  : P(rightward curvature)   in [0,1]
        t  : trunk-shift signed score in R
        zc : shoulder-side logit      in R
    """
    return {
        "xray_advised": abs(2 * readout["g"] - 1),
        "curve_side":   abs(2 * readout["d"] - 1),
        "trunk":        min(abs(readout["t"]) / TRUNK_SCALE, 1.0),
        "shoulder":     min(abs(readout["zc"]) / SHOULDER_SCALE, 1.0),
    }


def format_evidence(readout: dict) -> str:
    """Serialize the reliability-annotated body-surface evidence (the propagated Turn-1 text)."""
    g, d, t, zc = readout["g"], readout["d"], readout["t"], readout["zc"]
    advised = g >= 0.5
    r = concept_reliability(readout)
    lines = ["BODY-SURFACE EVIDENCE", "", "Screening:",
             ("Radiographic examination is recommended." if advised
              else "Radiographic examination is not indicated.")
             + f" Confidence: {confidence_word(r['xray_advised'])}.", ""]
    if advised:
        lines += ["Curve tendency:",
                  f"Body-surface morphology supports a {'rightward' if d >= 0.5 else 'leftward'} "
                  f"curvature tendency. Confidence: {confidence_word(r['curve_side'])}.", ""]
    if t > TAU_TRUNK_SIGN or t < -TAU_TRUNK_SIGN:
        side = "rightward" if t > 0 else "leftward"
        lines += ["Trunk:", f"A {side} trunk asymmetry is observed. "
                  f"Confidence: {confidence_word(r['trunk'])}.", ""]
    else:
        lines += ["Trunk:", "No clear trunk asymmetry.", ""]
    if abs(zc) < SHOULDER_ABSTAIN:                       # abstain when unreliable
        lines += ["Shoulder:", "Laterality evidence is insufficiently reliable. Status: Abstain."]
    else:
        lines += ["Shoulder:", f"{'Left' if zc > 0 else 'Right'}-sided shoulder elevation evidence. "
                  f"Confidence: {confidence_word(r['shoulder'])}."]
    return "\n".join(lines)
