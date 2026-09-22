"""LSEM — Latent Spatial Evidence Module.

Two stages:
  1. ``LSEMExtractor`` : frozen VLM -> mid-layer latent spatial evidence grid (2, R, C, D).
  2. ``LSEMReadout``   : structure-regularized weight-field readout over the intact grid.
"""
from .extract import LSEMExtractor, load_swift_vlm
from .readout import LSEMReadout

__all__ = ["LSEMExtractor", "load_swift_vlm", "LSEMReadout"]
__version__ = "0.1.0"
