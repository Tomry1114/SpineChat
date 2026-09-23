"""Minimal LSEM usage (no data shipped — supply your own image paths / labels).

Pipeline:  RGB+depth images --LSEMExtractor--> latent spatial grid --LSEMReadout--> concept logits.
"""
import torch
from lsem import LSEMExtractor, LSEMReadout, load_swift_vlm

SYSTEM = "You are a scoliosis screening assistant. Screen from the RGB-D back-surface images."
USER = ("<image><image>\nThese are the back-surface images (image 1: RGB, image 2: depth). "
        "Report the shoulder balance, the trunk shift, and the dominant lateral tendency.")

# ---- 1. frozen VLM + extractor (adapter optional) --------------------------------------------
model, template = load_swift_vlm("Qwen/Qwen3.5-4B", adapter="")          # your fine-tuned adapter dir
extractor = LSEMExtractor(model, template, layer=16, grid=(6, 4))

grid = extractor.extract("subject_rgb.png", "subject_depth.png", SYSTEM, USER)   # (2, 6, 4, D)
D = grid.shape[-1]

# ---- 2. readout (grid_fused config by default) -----------------------------------------------
concepts = {"lateral": 2, "shoulder": 3, "trunk": 3}                      # {concept: n_classes}
readout = LSEMReadout(in_dim=D, num_classes=concepts, grid_shape=(2, 6, 4),
                      lambda_group=0.01, lambda_fused=0.01, lambda_bilateral=0.0)

logits = readout(grid.unsqueeze(0), "lateral")                           # (1, 2)
print("lateral logits:", logits)
print("lateral region map (2x6x4):\n", readout.region_map("lateral"))
print("lateral bilateral (sym, anti):", readout.bilateral_energy("lateral"))

# ---- 3. training sketch (your loop; extract grids once, then train the readout) --------------
# loss = sum_c CrossEntropy(readout(grid_batch, c), y_c) + readout.structure_penalty(c)
