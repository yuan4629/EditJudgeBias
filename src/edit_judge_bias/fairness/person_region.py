"""Deterministic person / face / head / skin region extraction for the D-class track.

★ WHY THIS MODULE EXISTS

The 2026-07-30 D-class gate failed for one reason: `gpt-image-2` re-renders the whole
frame, so "change only the person's gender" also changed shoes, shirt colour, hand pose,
nail polish, sleeves and background texture. Two MLLM auditors then agreed at CHANCE on
whether a pair was usable (Cohen's κ = −0.028), and the divergence sat exactly on
`scene_preserved` (κ = +0.130) — the sub-question the protocol's validity rests on.

The fix is not a better prompt. It is to move locality from a *request* ("Change nothing
else", which the editor ignores) to a *construction*: every attribute change is composited
through a mask, so only masked pixels can differ, and `scene_preserved` becomes the
integer `max|out − base| outside the mask == 0` instead of an MLLM judgment. This module
produces those masks.

★ THREE MEASUREMENTS THAT SHAPED THE API (2026-07-30, all on the project's own pool)

1. **A skin-colour rule alone is unusable.** The classic YCrCb∩HSV skin test fires on sand
   and wood: over 57 candidate images the median share of the WHOLE FRAME it selected was
   **0.285**, max 0.994. So `skin_mask` takes a person prior and `require_person` defaults
   to True — for study rows it must never be False.
2. **Haar face detection alone is unusable.** Its largest "faces" in the pool were a
   **burger** (0.629 of frame) and a **TV remote** (0.473). So `face_boxes` filters every
   candidate by its overlap with the person mask (`min_person_overlap`, default 0.5).
3. **`content_category` is unreliable** — of the 139 originals with a face >2% of frame,
   only 87 were labelled `human`; the other 52 were global / object / scenery. The pool
   also contains gulls, a squirrel, a zebra and dolls. So personhood is a *measured*
   property here, never a manifest field.

★ THE CACHED PNG IS THE SOURCE OF TRUTH, NOT THE MODEL

DeepLabV3 argmax can flip boundary pixels between CUDA and CPU, so a mask recomputed at
gate time could differ from the one used at injection time and the byte-identity assertion
would fail for a reason that has nothing to do with the injection. Every mask is therefore
persisted as an 8-bit PNG and re-read; `cache_path_for` mixes a digest of the region config
into the filename, so changing e.g. `infer_long_edge` cannot silently reuse a stale mask.

★ INSTRUMENTS MUST PROVE THEMSELVES

`self_check` runs known-answer controls and RAISES under `strict=True`. This is a direct
descendant of a real incident: the ORB near-duplicate check once returned "0 duplicates"
because it was called with paths instead of signatures and the `TypeError` was swallowed by
an `except Exception: continue`. An instrument that fails silently is worse than none, so
`person_mask` raises `RegionError` on an unreadable image rather than returning an empty
mask — an empty mask would sail through every downstream check as "nothing to change".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Bbox = Tuple[int, int, int, int]  # (left, top, right, bottom); right/bottom exclusive,
                                  # the same convention as bias/edit_region.py

#: Pascal-VOC class index for "person" in torchvision's segmentation models.
PERSON_CLASS_VOC = 15

#: Default region config. Every key is surfaced in a YAML config; the values here are what
#: the tests and `self_check` run against.
DEFAULT_REGION_CONFIG: Dict[str, Any] = {
    "device": "auto",                     # auto | cuda | cpu
    "backend": "deeplabv3_resnet50",      # deeplabv3_resnet50 | deeplabv3_resnet101 | haar_grabcut
    "infer_long_edge": 640,
    "person_prob_threshold": 0.5,
    "morph_frac": 0.005,
    "face": {
        "cascades": ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"),
        # Normalised detection resolution -- see `_raw_face_boxes`. Matches the person mask's
        # `infer_long_edge` so both instruments see the same scale regime.
        "detect_long_edge": 640,
        "scale_factor": 1.1,
        "min_neighbors": 6,
        "min_size_frac": 0.04,
        "min_person_overlap": 0.5,
    },
    "head": {"grow_up": 0.9, "grow_down": 0.7, "grow_side": 0.45},
    "skin": {
        "cr": (133, 173),
        "cb": (77, 127),
        "y_min": 40,
        "s": (25, 190),
        "hue_wrap": 25,      # accept H <= 25 or H >= 180 - 10 on OpenCV's 0..179 scale
        "require_person": True,
    },
    "feather": {"radius_frac": 0.01},
    "seed": 42,
}


class RegionError(RuntimeError):
    """An instrument failed.

    Never caught and converted into an empty mask: an empty mask is indistinguishable
    from "this image has no person", which would let a broken instrument quietly shrink
    the pool instead of failing the run.
    """


@dataclass(frozen=True)
class InstrumentCheck:
    """One known-answer control over an instrument."""

    name: str
    passed: bool
    measured: float
    expected: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        flag = "ok  " if self.passed else "FAIL"
        return f"[{flag}] {self.name}: measured={self.measured:.4f} expected={self.expected} {self.detail}"


@dataclass(frozen=True)
class RegionSet:
    """Every region derived from one image, plus how it was derived."""

    image_path: Optional[Path]
    width: int
    height: int
    person: Any                  # np.ndarray bool (H, W)
    person_method: str           # "deeplabv3_resnet50" | ... | "haar_grabcut"
    person_frac: float
    face_box: Optional[Bbox]
    face_frac: float
    face_source: str             # cascade filename | "none"
    head: Any                    # np.ndarray bool
    head_frac: float
    skin: Any                    # np.ndarray bool
    skin_frac: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Lazy heavy imports — mirrors data/overlap.py::_cv2 so the 649-test suite      #
# keeps passing in a Pillow-only environment.                                  #
# --------------------------------------------------------------------------- #
def _np():
    try:
        import numpy as np  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RegionError("person-region extraction needs numpy (pip install numpy)") from exc
    return np


def _cv2():
    try:
        import cv2  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RegionError(
            "person-region extraction needs opencv-python (pip install opencv-python)"
        ) from exc
    return cv2


def _torch():
    """torch + torchvision, or None — the caller falls back to `haar_grabcut`.

    Deliberately returns None instead of raising: a missing segmenter is a *degraded*
    instrument, not a broken one, and which one ran is recorded in `person_method` so no
    row is ever ambiguous about how its mask was made.
    """
    try:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415
    except ImportError:  # pragma: no cover - environment-dependent
        return None
    return torch, torchvision


# --------------------------------------------------------------------------- #
# Config digest + cache paths                                                  #
# --------------------------------------------------------------------------- #
def _canonical(cfg: Any) -> Any:
    """Config as sorted, JSON-safe primitives so its digest is stable across runs."""
    if isinstance(cfg, dict):
        return {k: _canonical(cfg[k]) for k in sorted(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [_canonical(v) for v in cfg]
    if isinstance(cfg, Path):
        return cfg.as_posix()
    return cfg


def config_digest(cfg: Dict[str, Any]) -> str:
    """Short digest of the region config.

    Mixed into every cache filename. Without it, raising `infer_long_edge` would silently
    reuse masks computed at the old resolution — the same class of stale-artifact bug as
    the pilot/v2 `result_id` collision.
    """
    blob = json.dumps(_canonical(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def cache_path_for(image_path, kind: str, *, cache_dir: Path, digest: str) -> Path:
    """`<cache_dir>/<kind>/<digest>/<flattened image path>.png`.

    The image path is flattened rather than basenamed: the pool draws from several trees
    that reuse filenames (`model00/H_00_05.jpg` exists under both `sourceimg_h` and
    `targetimg_h`), so a basename key would collide across sources.
    """
    flat = Path(image_path).as_posix().replace("/", "__").replace(":", "_")
    return Path(cache_dir) / kind / digest / f"{flat}.png"


def mask_sha256(mask) -> str:
    """Digest of a boolean mask, for proving two arms shared one mask."""
    np = _np()
    arr = np.ascontiguousarray(np.asarray(mask).astype(bool))
    return hashlib.sha256(arr.tobytes() + repr(arr.shape).encode()).hexdigest()[:16]


def save_mask_png(mask, path: Path) -> Path:
    """Persist a boolean mask as an 8-bit PNG of {0, 255}."""
    from PIL import Image

    np = _np()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask).astype(bool).astype("uint8")) * 255
    Image.fromarray(arr, mode="L").save(path)
    return path


def load_mask_png(path: Path):
    """Re-read a persisted mask. The PNG is the source of truth, not the model."""
    from PIL import Image

    np = _np()
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


# --------------------------------------------------------------------------- #
# Person mask                                                                  #
# --------------------------------------------------------------------------- #
def _as_rgb_array(image) -> Any:
    """(H, W, 3) uint8 RGB from a path, a PIL image, or an array. Raises on failure."""
    from PIL import Image

    np = _np()
    if image is None:
        raise RegionError("no image given")
    if hasattr(image, "shape"):
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RegionError(f"expected an (H, W, 3) RGB array, got shape {arr.shape}")
        return np.ascontiguousarray(arr[:, :, :3].astype("uint8"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    try:
        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as exc:
        # ★ Raise, never return an empty mask. See the module docstring.
        raise RegionError(f"unreadable image {image!r}: {type(exc).__name__}: {exc}") from exc


def _morph_clean(mask, *, short_edge: int, morph_frac: float):
    """Open then close, with a kernel that is a constant fraction of the short edge.

    Same reasoning as `bias/base.py::relative_px`: the pool mixes 500, 512 and 1024 px
    frames, so a fixed-pixel kernel would clean a 512 px image differently from a 1024 px
    one and make region size partly a function of resolution.
    """
    np = _np()
    cv2 = _cv2()
    k = max(1, int(round(short_edge * morph_frac)))
    if k <= 1:
        return mask
    kernel = np.ones((k, k), np.uint8)
    out = mask.astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(bool)


_DEEPLAB_STATE: Dict[str, Any] = {}


def _person_mask_deeplab(rgb, *, cfg: Dict[str, Any]) -> Optional[Any]:
    """Person mask from torchvision DeepLabV3, or None if torch/weights are unavailable."""
    mods = _torch()
    if mods is None:
        return None
    torch, torchvision = mods
    np = _np()
    backend = str(cfg.get("backend", "deeplabv3_resnet50"))
    if not backend.startswith("deeplabv3"):
        return None

    want = str(cfg.get("device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want

    key = f"{backend}:{device}"
    model = _DEEPLAB_STATE.get(key)
    if model is None:
        try:
            ctor = getattr(torchvision.models.segmentation, backend)
            model = ctor(weights="DEFAULT")
        except Exception as exc:  # pragma: no cover - needs a weights download
            raise RegionError(
                f"could not build {backend} with pretrained weights: {type(exc).__name__}: {exc}. "
                "Either allow the one-off torchvision weights download, or set "
                "backend: haar_grabcut in the region config."
            ) from exc
        model.eval().to(device)
        # Determinism: cudnn autotuning can pick different kernels per run, which moves
        # boundary pixels. The cached PNG protects us anyway, but do not add noise.
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover
            pass
        _DEEPLAB_STATE[key] = model

    h, w = rgb.shape[:2]
    long_edge = int(cfg.get("infer_long_edge", 640))
    scale = long_edge / max(h, w) if max(h, w) > long_edge else 1.0
    if scale != 1.0:
        cv2 = _cv2()
        small = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    x = ((small.astype("float32") / 255.0) - mean) / std
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)["out"][0]
        probs = torch.softmax(logits, dim=0)[PERSON_CLASS_VOC]
    small_mask = (probs.detach().cpu().numpy() >= float(cfg.get("person_prob_threshold", 0.5)))

    if small_mask.shape != (h, w):
        cv2 = _cv2()
        small_mask = cv2.resize(small_mask.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    return _morph_clean(small_mask, short_edge=min(h, w),
                        morph_frac=float(cfg.get("morph_frac", 0.005)))


def _person_mask_haar_grabcut(rgb, *, cfg: Dict[str, Any]) -> Any:
    """Fallback: Haar face box -> generous head/torso box -> grabCut refinement.

    Weaker than DeepLabV3 and it knows it: `person_method` records which one ran so a row
    can never be ambiguous about the provenance of its mask.
    """
    np = _np()
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    boxes = _raw_face_boxes(rgb, cfg=cfg)
    if not boxes:
        return np.zeros((h, w), dtype=bool)
    l, t, r, b = max(boxes, key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]))
    fw, fh = r - l, b - t
    rect = (
        max(0, int(l - 1.2 * fw)),
        max(0, int(t - 0.8 * fh)),
        min(w, int(r + 1.2 * fw)),
        min(h, int(b + 4.0 * fh)),
    )
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    if rw < 4 or rh < 4:
        return np.zeros((h, w), dtype=bool)
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, gc, (rect[0], rect[1], rw, rh), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception as exc:
        raise RegionError(f"grabCut failed: {type(exc).__name__}: {exc}") from exc
    mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return _morph_clean(mask, short_edge=min(h, w), morph_frac=float(cfg.get("morph_frac", 0.005)))


def person_mask(image, *, cfg: Optional[Dict[str, Any]] = None,
                cache_dir: Optional[Path] = None) -> Tuple[Any, str, Dict[str, Any]]:
    """(bool mask, method, diagnostics). Raises `RegionError` on an unreadable image."""
    cfg = _merge_cfg(cfg)
    rgb = _as_rgb_array(image)
    digest = config_digest(cfg)
    diagnostics: Dict[str, Any] = {"config_digest": digest}

    cached: Optional[Path] = None
    if cache_dir is not None and not hasattr(image, "shape"):
        from PIL import Image as _PILImage
        if not isinstance(image, _PILImage.Image):
            cached = cache_path_for(image, "person", cache_dir=Path(cache_dir), digest=digest)
            if cached.exists():
                diagnostics.update({"cache": "hit", "cache_path": cached.as_posix(),
                                    "method": "cached_png"})
                # The method that originally produced it is recorded in the sidecar.
                side = cached.with_suffix(".json")
                method = "cached_png"
                if side.exists():
                    try:
                        method = json.loads(side.read_text(encoding="utf-8")).get("method", method)
                    except Exception:  # pragma: no cover
                        pass
                return load_mask_png(cached), method, diagnostics

    backend = str(cfg.get("backend", "deeplabv3_resnet50"))
    mask = None
    method = "haar_grabcut"
    if backend.startswith("deeplabv3"):
        mask = _person_mask_deeplab(rgb, cfg=cfg)
        if mask is not None:
            method = backend
    if mask is None:
        mask = _person_mask_haar_grabcut(rgb, cfg=cfg)
        method = "haar_grabcut"

    mods = _torch()
    if mods is not None:
        torch, torchvision = mods
        diagnostics.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "cuda_available": bool(torch.cuda.is_available())})
    diagnostics["method"] = method

    if cached is not None:
        save_mask_png(mask, cached)
        cached.with_suffix(".json").write_text(
            json.dumps({"method": method, "config_digest": digest}), encoding="utf-8"
        )
        diagnostics.update({"cache": "miss", "cache_path": cached.as_posix()})
        # ★ Round-trip immediately so every consumer sees the SAME bytes the gate will,
        # regardless of CUDA/CPU boundary-pixel differences.
        mask = load_mask_png(cached)
    return mask, method, diagnostics


# --------------------------------------------------------------------------- #
# Face boxes                                                                   #
# --------------------------------------------------------------------------- #
def _raw_face_boxes(rgb, *, cfg: Dict[str, Any]) -> List[Bbox]:
    """Unfiltered Haar detections. Not for direct use — see `face_boxes`.

    ★ DETECTION RUNS AT A NORMALISED RESOLUTION, and this is a correctness fix, not just a
    speed one. Haar's sensitivity depends on absolute pixel size, so screening EBench (512x512)
    and OmniEdit (768x1344) at native resolution puts the two corpora under *different detector
    regimes* -- which would bias the very cross-corpus yield comparison the probe exists to make.
    `min_size_frac` keeps the minimum relative, but the detector's behaviour above that floor
    still varies with absolute scale.

    Detections are scaled back to native coordinates, so every consumer still sees native-frame
    boxes. Measured side benefit: Haar was 97 ms of a 231 ms budget per 1344px image (42%);
    normalising to a 640px long edge cuts that roughly fourfold.
    """
    cv2 = _cv2()
    face_cfg = cfg["face"]
    h, w = rgb.shape[:2]
    long_edge = int(face_cfg.get("detect_long_edge", 640))
    scale = long_edge / max(h, w) if long_edge and max(h, w) > long_edge else 1.0
    if scale != 1.0:
        small = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    sh, sw = small.shape[:2]
    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    min_side = max(12, int(round(min(sh, sw) * float(face_cfg.get("min_size_frac", 0.04)))))
    out: List[Bbox] = []
    for name in face_cfg.get("cascades", ()):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():  # pragma: no cover - broken opencv install
            raise RegionError(f"could not load Haar cascade {name!r}")
        for (x, y, bw, bh) in cascade.detectMultiScale(
            grey,
            float(face_cfg.get("scale_factor", 1.1)),
            int(face_cfg.get("min_neighbors", 6)),
            minSize=(min_side, min_side),
        ):
            # Back to native coordinates so callers never see detector-space boxes.
            out.append((int(x / scale), int(y / scale),
                        int((x + bw) / scale), int((y + bh) / scale)))
    return out


def face_boxes(rgb, *, person=None, cfg: Optional[Dict[str, Any]] = None
               ) -> List[Tuple[Bbox, float]]:
    """[(box, person_overlap)] sorted largest first, filtered by the person mask.

    ★ The filter is load-bearing. Measured on this project's pool, Haar's largest "faces"
    were a burger (0.629 of frame) and a TV remote (0.473). Admitting those is exactly how
    the previous D attempt ended up judging gulls, a squirrel, a zebra and dolls. Do not
    lower `min_person_overlap` to grow n.
    """
    cfg = _merge_cfg(cfg)
    np = _np()
    rgb = _as_rgb_array(rgb)
    threshold = float(cfg["face"].get("min_person_overlap", 0.5))
    out: List[Tuple[Bbox, float]] = []
    for box in _raw_face_boxes(rgb, cfg=cfg):
        l, t, r, b = box
        if person is None:
            overlap = 1.0
        else:
            window = np.asarray(person)[t:b, l:r]
            overlap = float(window.mean()) if window.size else 0.0
        if overlap >= threshold:
            out.append((box, overlap))
    out.sort(key=lambda pair: (pair[0][2] - pair[0][0]) * (pair[0][3] - pair[0][1]), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Head / skin / non-skin regions                                              #
# --------------------------------------------------------------------------- #
def head_region(face: Bbox, person, *, cfg: Optional[Dict[str, Any]] = None):
    """Face box grown for hair/neck, intersected with the person mask."""
    cfg = _merge_cfg(cfg)
    np = _np()
    person = np.asarray(person).astype(bool)
    h, w = person.shape[:2]
    l, t, r, b = face
    fw, fh = r - l, b - t
    grow = cfg["head"]
    box = (
        max(0, int(round(l - fw * float(grow.get("grow_side", 0.45))))),
        max(0, int(round(t - fh * float(grow.get("grow_up", 0.9))))),
        min(w, int(round(r + fw * float(grow.get("grow_side", 0.45))))),
        min(h, int(round(b + fh * float(grow.get("grow_down", 0.7))))),
    )
    window = np.zeros((h, w), dtype=bool)
    window[box[1]:box[3], box[0]:box[2]] = True
    return window & person


def skin_mask(rgb, person=None, *, cfg: Optional[Dict[str, Any]] = None,
              require_person: Optional[bool] = None) -> Tuple[Any, Dict[str, Any]]:
    """Chroma-rule skin pixels, intersected with the person mask.

    ★ `require_person` must stay True for study rows. Measured on 57 candidate images, the
    bare chroma rule selected a median **0.285** of the WHOLE FRAME (max 0.994) because
    sand and wood are skin-coloured. Recolouring sand is not a skin-tone manipulation.

    `require_person=False` exists for exactly one legitimate use: harvesting a
    skin-*like* NON-person region as an artefact control, where the over-fire is the point.
    """
    cfg = _merge_cfg(cfg)
    np = _np()
    cv2 = _cv2()
    rgb = _as_rgb_array(rgb)
    skin_cfg = cfg["skin"]
    if require_person is None:
        require_person = bool(skin_cfg.get("require_person", True))
    if require_person and person is None:
        raise RegionError(
            "skin_mask(require_person=True) needs a person mask; pass one, or pass "
            "require_person=False only for the non-person artefact control"
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycc)
    hue, sat, _val = cv2.split(hsv)
    cr_lo, cr_hi = skin_cfg.get("cr", (133, 173))
    cb_lo, cb_hi = skin_cfg.get("cb", (77, 127))
    s_lo, s_hi = skin_cfg.get("s", (25, 190))
    wrap = int(skin_cfg.get("hue_wrap", 25))
    chroma = (
        (cr >= cr_lo) & (cr <= cr_hi)
        & (cb >= cb_lo) & (cb <= cb_hi)
        & (y > int(skin_cfg.get("y_min", 40)))
        & (sat >= s_lo) & (sat <= s_hi)
        & ((hue <= wrap) | (hue >= 180 - 10))
    )
    stats = {"chroma_frac": float(chroma.mean()), "require_person": bool(require_person)}
    mask = chroma
    if require_person:
        mask = chroma & np.asarray(person).astype(bool)
    mask = _morph_clean(mask, short_edge=min(rgb.shape[:2]),
                        morph_frac=float(cfg.get("morph_frac", 0.005)))
    stats["skin_frac"] = float(mask.mean())
    return mask, stats


def nonskin_same_person_region(person, skin, *, target_area: int,
                               cfg: Optional[Dict[str, Any]] = None
                               ) -> Tuple[Optional[Any], Dict[str, Any]]:
    """An area-matched NON-skin region on the SAME person (clothing / hair).

    ★ Why the same person and not "any non-person region of equal area": equal *area* is
    not equal *saliency*. Recolouring a wall is less noticed than recolouring a person, so
    a background control estimates only a LOWER BOUND on the artefact penalty, and
    subtracting a lower bound inflates the residual one would then call bias — an error in
    the unsafe direction. Clothing and hair on the same subject match saliency, seam
    physics and object identity, so they isolate "the judge dislikes recoloured pixels"
    from "the judge dislikes this skin tone".
    """
    np = _np()
    cv2 = _cv2()
    person = np.asarray(person).astype(bool)
    skin = np.asarray(skin).astype(bool)
    candidate = person & ~skin
    stats: Dict[str, Any] = {"candidate_area": int(candidate.sum()),
                             "target_area": int(target_area)}
    if target_area <= 0 or not candidate.any():
        stats["reason"] = "no_nonskin_region_on_person"
        return None, stats

    labels_n, labels = cv2.connectedComponents(candidate.astype(np.uint8))
    best, best_area = None, 0
    for idx in range(1, labels_n):
        area = int((labels == idx).sum())
        if area > best_area:
            best, best_area = idx, area
    region = labels == best
    # Grow or shrink toward the target area so the two arms carry a matched pixel budget.
    kernel = np.ones((3, 3), np.uint8)
    guard = 0
    while region.sum() < target_area * 0.8 and guard < 64:
        grown = cv2.dilate(region.astype(np.uint8), kernel, iterations=1).astype(bool) & candidate
        if grown.sum() <= region.sum():
            break
        region, guard = grown, guard + 1
    if region.sum() > target_area * 1.25:
        ys, xs = np.where(region)
        order = np.lexsort((xs, ys))          # deterministic: row-major
        keep = order[:int(target_area)]
        trimmed = np.zeros_like(region)
        trimmed[ys[keep], xs[keep]] = True
        region = trimmed

    achieved = int(region.sum())
    stats.update({"achieved_area": achieved,
                  "area_ratio": (achieved / target_area) if target_area else None,
                  "overlaps_skin": int((region & skin).sum())})
    if not 0.8 <= stats["area_ratio"] <= 1.25:
        stats["reason"] = "area_match_out_of_bounds"
        return None, stats
    return region, stats


def feather(mask, *, radius_frac: float, short_edge: int):
    """Soft alpha in [0, 1]: 1 on an eroded core, 0 outside a dilated boundary.

    `alpha > 0` is a strict superset of `mask`, so the byte-identity assertion must be
    made against `alpha > 0`, never against `mask` — otherwise the feather band would be
    outside the "unchanged" region while actually being modified.
    """
    np = _np()
    cv2 = _cv2()
    mask = np.asarray(mask).astype(bool)
    r = max(0, int(round(short_edge * float(radius_frac))))
    if r == 0 or not mask.any():
        return mask.astype("float32")
    k = 2 * r + 1
    blurred = cv2.GaussianBlur(mask.astype("float32"), (k, k), 0)
    kernel = np.ones((k, k), np.uint8)
    core = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    support = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    alpha = np.clip(blurred, 0.0, 1.0)
    alpha[core] = 1.0
    alpha[~support] = 0.0
    return alpha.astype("float32")


def outside_mask_identical(base, out, alpha) -> Tuple[bool, int]:
    """(is byte-identical outside `alpha > 0`, max abs diff there).

    ★ This single integer replaces the MLLM `scene_preserved` question, on which two
    independent auditors reached κ = +0.130 while agreeing at κ = +0.716 on the easy
    "is a person present". Locality is now arithmetic.
    """
    np = _np()
    a = np.asarray(base).astype(np.int16)
    b = np.asarray(out).astype(np.int16)
    if a.shape != b.shape:
        raise RegionError(f"shape mismatch: base {a.shape} vs out {b.shape}")
    support = np.asarray(alpha) > 0
    if support.shape != a.shape[:2]:
        raise RegionError(f"alpha shape {support.shape} does not match image {a.shape[:2]}")
    diff = np.abs(a - b).max(axis=2) if a.ndim == 3 else np.abs(a - b)
    outside = diff[~support]
    worst = int(outside.max()) if outside.size else 0
    return worst == 0, worst


# --------------------------------------------------------------------------- #
# The bundle                                                                   #
# --------------------------------------------------------------------------- #
def _merge_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge a user config over the defaults, one level into each sub-dict."""
    merged = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in DEFAULT_REGION_CONFIG.items()}
    for key, value in (cfg or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def region_set(image, *, cfg: Optional[Dict[str, Any]] = None,
               cache_dir: Optional[Path] = None) -> RegionSet:
    """Every region for one image, in one pass."""
    cfg = _merge_cfg(cfg)
    np = _np()
    rgb = _as_rgb_array(image)
    h, w = rgb.shape[:2]

    person, method, diagnostics = person_mask(image, cfg=cfg, cache_dir=cache_dir)
    faces = face_boxes(rgb, person=person, cfg=cfg)
    face_box = faces[0][0] if faces else None
    if face_box is not None:
        l, t, r, b = face_box
        face_frac = ((r - l) * (b - t)) / float(max(1, w * h))
        head = head_region(face_box, person, cfg=cfg)
        face_source = "haar_person_filtered"
    else:
        face_frac, head, face_source = 0.0, np.zeros((h, w), dtype=bool), "none"

    skin, skin_stats = skin_mask(rgb, person, cfg=cfg)
    diagnostics.update({"skin": skin_stats, "n_face_candidates": len(faces),
                        "face_overlap": faces[0][1] if faces else None})

    return RegionSet(
        image_path=None if hasattr(image, "shape") else Path(str(image)),
        width=w, height=h,
        person=person, person_method=method, person_frac=float(person.mean()),
        face_box=face_box, face_frac=face_frac, face_source=face_source,
        head=head, head_frac=float(head.mean()),
        skin=skin, skin_frac=float(skin.mean()),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Known-answer self-validation                                                 #
# --------------------------------------------------------------------------- #
def self_check(*, cfg: Optional[Dict[str, Any]] = None, strict: bool = True
               ) -> List[InstrumentCheck]:
    """Run synthetic known-answer controls. Raises `RegionError` on failure when strict.

    Synthetic-only so it runs in CI with no data tree and no model weights; the pool-level
    checks live in `self_check_on_pool`.
    """
    np = _np()
    cfg = _merge_cfg(cfg)
    checks: List[InstrumentCheck] = []

    # 1. A pure skin-chroma frame with a 10% person: the person prior must bound the mask.
    skinny = np.zeros((100, 100, 3), dtype="uint8")
    skinny[:, :] = (200, 150, 130)                      # skin-like everywhere
    person = np.zeros((100, 100), dtype=bool)
    person[:10, :] = True                               # 10% of the frame
    bounded, _ = skin_mask(skinny, person, cfg=cfg)
    checks.append(InstrumentCheck(
        "skin_needs_person", bool(bounded.mean() <= 0.10 + 1e-9), float(bounded.mean()),
        "<= 0.10", "the person prior must bound a frame-wide chroma hit",
    ))
    unbounded, _ = skin_mask(skinny, None, cfg=cfg, require_person=False)
    checks.append(InstrumentCheck(
        "skin_without_person_overfires", bool(unbounded.mean() >= 0.90),
        float(unbounded.mean()), ">= 0.90",
        "documents the 0.285-median over-fire the person prior exists to fix",
    ))

    # 2. The person-overlap filter must reject a detection that lies off the person.
    #    Synthetic Haar positives are unreliable, so drive `face_boxes` with a stub box by
    #    checking the overlap arithmetic directly on a known mask.
    off = np.zeros((100, 100), dtype=bool)
    off[0:10, 0:10] = True
    window = off[50:70, 50:70]
    checks.append(InstrumentCheck(
        "face_overlap_filter_arithmetic", bool(window.mean() == 0.0), float(window.mean()),
        "== 0.0", "a box off the person must score 0 overlap and be dropped",
    ))

    # 3. Feather must partition the frame: core == 1, outside support == 0, support ⊇ mask.
    m = np.zeros((80, 80), dtype=bool)
    m[30:50, 30:50] = True
    alpha = feather(m, radius_frac=0.05, short_edge=80)
    superset = bool(((alpha > 0) | ~m).all())
    checks.append(InstrumentCheck(
        "feather_support_contains_mask", superset, float((alpha > 0).mean()),
        "alpha>0 superset of mask", "byte-identity must be asserted against alpha>0",
    ))
    checks.append(InstrumentCheck(
        "feather_is_bounded", bool(alpha.min() >= 0.0 and alpha.max() <= 1.0),
        float(alpha.max()), "within [0, 1]",
    ))

    # 4. Mask PNG round-trip must be bit-identical, and the digest must follow the config.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = save_mask_png(m, Path(tmp) / "m.png")
        again = load_mask_png(path)
        checks.append(InstrumentCheck(
            "cache_roundtrip_bit_identical", bool((again == m).all()),
            float((again == m).mean()), "== 1.0",
        ))
    d1 = config_digest(cfg)
    d2 = config_digest({**cfg, "infer_long_edge": int(cfg["infer_long_edge"]) + 1})
    checks.append(InstrumentCheck(
        "config_digest_tracks_config", bool(d1 != d2), 1.0 if d1 != d2 else 0.0,
        "different", "a changed config must not reuse a cached mask",
    ))

    # 5. outside_mask_identical must catch a single stray pixel.
    base = np.zeros((40, 40, 3), dtype="uint8")
    out = base.copy()
    out[0, 0] = (1, 0, 0)                     # one LSB outside the mask
    inside = np.zeros((40, 40), dtype=bool)
    inside[20:30, 20:30] = True
    ok, worst = outside_mask_identical(base, out, inside)
    checks.append(InstrumentCheck(
        "outside_identity_catches_one_pixel", bool(ok is False and worst == 1), float(worst),
        "detected, max diff 1", "no tolerance: one stray pixel is a bug, not noise",
    ))

    # 6. An unreadable image must RAISE, not yield an empty mask.
    raised = False
    try:
        _as_rgb_array(Path("this-file-does-not-exist-9f3a.png"))
    except RegionError:
        raised = True
    checks.append(InstrumentCheck(
        "fails_loudly_on_unreadable", raised, 1.0 if raised else 0.0, "raises RegionError",
        "an empty mask would pass every downstream check as 'nothing to change'",
    ))

    if strict:
        failed = [c for c in checks if not c.passed]
        if failed:
            raise RegionError(
                "person_region self-check failed:\n" + "\n".join(str(c) for c in failed)
            )
    return checks


def self_check_on_pool(image_paths: Sequence, *, cfg: Optional[Dict[str, Any]] = None,
                       cache_dir: Optional[Path] = None, n: int = 8,
                       person_frac_bounds: Tuple[float, float] = (0.02, 0.98),
                       strict: bool = False) -> List[InstrumentCheck]:
    """Sanity-check the live backend on real images. Called by the pool builder, not pytest."""
    cfg = _merge_cfg(cfg)
    checks: List[InstrumentCheck] = []
    lo, hi = person_frac_bounds
    for path in list(image_paths)[:n]:
        rs = region_set(path, cfg=cfg, cache_dir=cache_dir)
        checks.append(InstrumentCheck(
            f"person_frac_plausible[{Path(str(path)).name}]",
            bool(lo <= rs.person_frac <= hi), rs.person_frac, f"within [{lo}, {hi}]",
            f"method={rs.person_method} face_frac={rs.face_frac:.4f}",
        ))
    if strict:
        failed = [c for c in checks if not c.passed]
        if failed:
            raise RegionError(
                "person_region pool self-check failed:\n" + "\n".join(str(c) for c in failed)
            )
    return checks


def _main() -> None:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="person_region instrument self-check")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--image", action="append", default=[],
                    help="also run the pool check on these images")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args()

    strict = not args.no_strict
    for check in self_check(strict=strict):
        print(check)
    if args.image:
        cache = Path(args.cache_dir) if args.cache_dir else None
        for check in self_check_on_pool(args.image, cache_dir=cache, n=len(args.image),
                                        strict=strict):
            print(check)


if __name__ == "__main__":  # pragma: no cover
    _main()
