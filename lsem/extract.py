"""LSEM latent spatial evidence extraction — frozen VLM -> mid-layer token grid.

Forward a frozen vision-language model on the two back-surface images (RGB + depth), read the
hidden states at a chosen mid layer, locate the RGB and depth image-token spans, reshape each to
its spatial token grid, and adaptive-average-pool each to a fixed ``(rows, cols)`` grid. The result
is a structure-preserving *latent spatial evidence* tensor of shape ``(2, rows, cols, D)`` that
feeds :class:`lsem.readout.LSEMReadout`.

The extractor is backbone-agnostic in spirit but the reference implementation targets a
Qwen-family VLM served through ms-swift (``image_grid_thw`` + a single image placeholder id).
Pass your own ``model``/``template`` and adjust ``img_token_id`` / ``merge`` for other backbones.
"""
from __future__ import annotations
import re
import numpy as np
import torch
import torch.nn.functional as F

# Default Qwen3.5-VL image placeholder id and spatial-merge factor; override for other backbones.
QWEN_IMG_TOKEN_ID = 248056
QWEN_MERGE = 2


def _runs(mask: np.ndarray):
    """Contiguous True-runs in a boolean array -> list of (start, end) inclusive."""
    idx = np.where(mask)[0]
    runs = []
    if len(idx):
        s = p = idx[0]
        for j in idx[1:]:
            if j == p + 1:
                p = j
            else:
                runs.append((int(s), int(p))); s = p = j
        runs.append((int(s), int(p)))
    return runs


class LSEMExtractor:
    """Extract the LSEM latent spatial grid from a frozen VLM.

    Parameters
    ----------
    model, template : a loaded (frozen) ms-swift VLM and its template (see ``load_swift_vlm``).
    layer : hidden layer to read (default 16; the paper reports mid-layer > deep-layer).
    grid : fixed (rows, cols) pooled grid per modality (default (6, 4) -> 2x6x4 cells).
    img_token_id, merge : backbone image placeholder id and spatial-merge factor.
    """

    def __init__(self, model, template, layer: int = 16, grid=(6, 4),
                 img_token_id: int = QWEN_IMG_TOKEN_ID, merge: int = QWEN_MERGE, device: str = "cuda"):
        self.model, self.template = model, template
        self.layer, self.grid, self.img_token_id, self.merge, self.device = layer, grid, img_token_id, merge, device
        self._feat = {}
        for name, mod in model.named_modules():
            m = re.match(r".*\.layers\.(\d+)$", name)
            if m and int(m.group(1)) == layer:
                mod.register_forward_hook(lambda _m, _i, o: self._feat.__setitem__(
                    "h", (o[0] if isinstance(o, tuple) else o).detach()))
                break

    def _pool(self, h: torch.Tensor, span, gh: int, gw: int) -> torch.Tensor:
        """Reshape one image-token span to its gh x gw grid and adaptive-pool to self.grid -> (R, C, D)."""
        t = h[span[0]:span[1] + 1].float()
        if gh * gw != t.shape[0]:                                    # robustness to token/grid mismatch
            gw = int(round(t.shape[0] ** 0.5)); gh = (t.shape[0] + gw - 1) // gw
            t = torch.cat([t, t[-1:].repeat(gh * gw - t.shape[0], 1)], 0)
        g = t.reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)      # 1 x D x gh x gw
        return F.adaptive_avg_pool2d(g, self.grid)[0].permute(1, 2, 0)  # R x C x D

    @torch.no_grad()
    def extract(self, rgb_path: str, depth_path: str, system_prompt: str, user_prompt: str) -> torch.Tensor:
        """Return the latent spatial grid (2, R, C, D) for one subject's RGB + depth back-surface images.

        ``user_prompt`` must contain two image placeholders (``<image><image>``) for RGB then depth.
        """
        rec = {"messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": "ok"}],
               "images": [rgb_path, depth_path]}
        enc = self.template.encode(rec)
        b = self.template.data_collator([enc])
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        self._feat.clear()
        self.model(**b, use_cache=False)
        h = self._feat["h"][0]
        ids = b["input_ids"][0]; thw = b["image_grid_thw"]
        rs = _runs((ids == self.img_token_id).cpu().numpy())
        rgb_span, depth_span = rs[0], rs[1]
        pr = self._pool(h, rgb_span, int(thw[0][1]) // self.merge, int(thw[0][2]) // self.merge)
        pd = self._pool(h, depth_span, int(thw[1][1]) // self.merge, int(thw[1][2]) // self.merge)
        return torch.stack([pr, pd], 0).cpu()                        # 2 x R x C x D


def load_swift_vlm(model_id: str, adapter: str = "", max_pixels: int = 1003520, device: str = "cuda"):
    """Convenience loader: a frozen ms-swift VLM (+ optional LoRA adapter) and its template."""
    from swift.model.register import get_model_processor
    from swift.template import get_template
    model, proc = get_model_processor(model_id, load_model=True)
    if adapter:
        from swift import Swift
        model = Swift.from_pretrained(model, adapter)
    model.eval(); model.config.use_cache = False; model.to(device)
    template = get_template(proc, max_pixels=max_pixels, max_length=4096); template.set_mode("train")
    return model, template
