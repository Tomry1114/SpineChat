#!/usr/bin/env python
"""SpineChat Stage 2 — train the LSEM readout heads on a FROZEN VLM's mid-layer grid.

Stage 2 is a probe on frozen features: no VLM gradients. First extract the latent spatial grids
with ``lsem.extract.LSEMExtractor`` (VLM frozen) and save them, then run this to train one
structure-regularized readout per concept:  L = Σ_c [ CE(readout(X, c), y_c) + Ω(W_c) ].

Expected --features .npz:
    X        : float32 (N, M, R, C, D)   latent grids (M=2 for RGB+depth), from LSEMExtractor
    y_<name> : int64   (N,)              per-concept labels (use -1 for missing/ignored)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from lsem import LSEMReadout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help=".npz with X and y_<concept> arrays")
    ap.add_argument("--concepts", nargs="+", required=True, help="e.g. lateral shoulder trunk xray")
    ap.add_argument("--out", default="lsem_readout.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    z = np.load(args.features)
    X = torch.tensor(z["X"], dtype=torch.float32)                       # N, M, R, C, D
    Y = {c: torch.tensor(z[f"y_{c}"], dtype=torch.long) for c in args.concepts}
    nclass = {c: int(Y[c][Y[c] >= 0].max().item()) + 1 for c in args.concepts}
    grid_shape = tuple(X.shape[1:4])

    net = LSEMReadout(in_dim=X.shape[-1], num_classes=nclass, grid_shape=grid_shape)
    opt = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        loss = X.new_zeros(())
        for c in args.concepts:
            loss = loss + ce(net(X, c), Y[c]) + net.structure_penalty(c)   # CE + Ω(W_c)
        loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"epoch {ep:4d}  loss {float(loss):.4f}", flush=True)

    torch.save({"state_dict": net.state_dict(), "num_classes": nclass, "grid_shape": grid_shape}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
