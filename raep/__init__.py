"""RAEP — Reliability-Aware Evidence Propagation.

Inference-time (no trainable parameters): attach a margin-derived reliability to each LSEM
screening concept, verbalize it, and propagate it in a fixed evidence order to the radiographic
diagnosis. ``SpineChatPipeline`` wires LSEM + CurveToken grounding + RAEP into one automated pass.
"""
from .reliability import concept_reliability, confidence_word, format_evidence
from .pipeline import SpineChatPipeline

__all__ = ["concept_reliability", "confidence_word", "format_evidence", "SpineChatPipeline"]
__version__ = "0.1.0"
