"""Automatic quality-preservation metrics.

SSIM between the original edited image and the biased image is always available
(pure skimage). CLIP image similarity and LPIPS are optional and load heavy
models lazily — they return None if their libraries/weights are unavailable, so
the pipeline degrades gracefully rather than failing.

Some biases enlarge the canvas rather than repaint it: `padding` adds a symmetric
border, `detail_caption` appends a caption strip below the frame. Both leave the
original pixels untouched, so SSIM must be measured on the *aligned* content. A
center-crop alone is wrong for the asymmetric case -- cropping a bottom-extended
image from the center shifts it vertically by half the strip height and drove
`detail_caption` to a spurious 0.46 (min 0.04) even though it changes no original
pixel. `compute_ssim` therefore removes the unknown translation by taking the best
SSIM over the plausible crop anchors. Misalignment is a measurement artifact of a
known canvas transform, not a quality loss; an image whose content really was
damaged scores low at every anchor, so maximising over anchors cannot manufacture
a passing score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim

_CLIP_STATE: dict = {}


def _gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _center_crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = arr.shape[:2]
    top, left = (H - h) // 2, (W - w) // 2
    return arr[top : top + h, left : left + w]


def _anchors(delta: int) -> tuple:
    """Candidate crop offsets along one axis: start, middle, end of the surplus."""
    return tuple(sorted({0, delta // 2, delta}))


def best_aligned_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM with the unknown canvas-extension offset marginalised out.

    `b` is cropped to `a`'s shape at every combination of the plausible vertical
    and horizontal anchors (at most 3x3), and the best score wins -- see the
    module docstring for why maximising here is sound.
    """
    if b.shape == a.shape:
        return float(_ssim(a, b, data_range=255.0))
    if b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1]:
        a, b = b, a  # compare against whichever is smaller
    if b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1]:
        # Neither contains the other (one axis grew, the other shrank); fall back
        # to a center-crop on both so the call still returns a number.
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        return float(_ssim(_center_crop(a, h, w), _center_crop(b, h, w), data_range=255.0))

    h, w = a.shape[:2]
    best = -1.0
    for top in _anchors(b.shape[0] - h):
        for left in _anchors(b.shape[1] - w):
            score = float(_ssim(a, b[top : top + h, left : left + w], data_range=255.0))
            if score > best:
                best = score
    return best


def compute_ssim(original_edited, biased) -> float:
    """SSIM in [−1, 1] (1 = identical structure), robust to canvas extension."""
    return best_aligned_ssim(_gray(original_edited), _gray(biased))


def compute_clip_similarity(original_edited, biased) -> Optional[float]:
    """Cosine similarity of CLIP image embeddings, or None if open_clip absent."""
    try:
        import open_clip  # noqa: F401
        import torch
    except ImportError:
        return None
    if "model" not in _CLIP_STATE:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        _CLIP_STATE.update(model=model, preprocess=preprocess, torch=torch)

    model, preprocess, torch = (
        _CLIP_STATE["model"], _CLIP_STATE["preprocess"], _CLIP_STATE["torch"]
    )
    with torch.no_grad():
        feats = []
        for p in (original_edited, biased):
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            f = model.encode_image(img)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return float((feats[0] * feats[1]).sum().item())


def compute_lpips(original_edited, biased) -> Optional[float]:
    """LPIPS perceptual distance, or None if the `lpips` package is unavailable."""
    try:
        import lpips  # type: ignore
        import torch
    except ImportError:
        return None
    if "lpips_fn" not in _CLIP_STATE:
        _CLIP_STATE["lpips_fn"] = lpips.LPIPS(net="alex")
        _CLIP_STATE["torch_l"] = torch
    fn, torch = _CLIP_STATE["lpips_fn"], _CLIP_STATE["torch_l"]

    def _t(p):
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    a, b = _t(original_edited), _t(biased)
    if a.shape != b.shape:
        b = b[:, :, : a.shape[2], : a.shape[3]]
    with torch.no_grad():
        return float(fn(a, b).item())


def compute_auto_metrics(
    original_edited,
    biased,
    *,
    with_clip: bool = False,
    with_lpips: bool = False,
) -> Dict[str, Optional[float]]:
    """SSIM (always) plus optional CLIP similarity and LPIPS distance."""
    out: Dict[str, Optional[float]] = {"ssim": compute_ssim(original_edited, biased)}
    out["clip_image_similarity"] = (
        compute_clip_similarity(original_edited, biased) if with_clip else None
    )
    out["lpips"] = compute_lpips(original_edited, biased) if with_lpips else None
    return out
