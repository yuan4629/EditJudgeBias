"""Content-pool overlap detection between a new source manifest and existing ones.

Enforces the data-integration rule that a new
source may only be called an independent content pool once the overlap has been
*measured*, not eyeballed. Until this existed the rule lived in human memory, which
is how a 96.6% instruction overlap (GenAI-Bench vs MagicBrush dev) got found by hand
rather than by a gate.

Two dimensions, because either one alone misses real cases:

- **pixels** — sha1 of the original-image bytes. Catches a source that literally
  redistributes someone else's inputs (ImagenHub-editing ⊂ MagicBrush dev).
- **text** — the normalised `instruction`. Catches a source that re-generated the
  images itself so every byte differs but the *content pool* is the same. This is
  the only dimension that would have caught GenAI-Bench.
- **near-duplicate** (opt-in, `--near-dup N`) — ORB keypoints + RANSAC homography
  between original images. Catches the case both of the above miss: the *same
  photograph*, cropped and re-encoded, under a different instruction. EBench-18K's
  low-level split is that case against I2EBench (both draw from GoPro / LOL / CSD),
  and it is the reason this dimension exists: perceptual hashes were measured to
  give **false negatives** there, because a crop moves every hash bucket.

Overlap is measured on **original** images only. Edited outputs are a function of
(original, instruction, editor), so two sources sharing an editor would look alike
without sharing any content — the pool is what the red line is about.

Overlap is reported in both directions: `*_overlap` is the share of the NEW source
found in the other one (the number the accept/reject thresholds apply to), and
`*_overlap_reverse` is the share of the other source found in the new one, which is
what tells a subset apart from a superset.

Transitivity is deliberately **not** automated — "source X says it used Y's inputs"
is a claim in a paper, not a field in a manifest. The report records what was
compared; chasing the chain (Emu Edit -> MagicBrush -> EditReward-Data ~70%) stays a
human step written into the provenance card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root, resolve
from edit_judge_bias.data.schema import SampleRecord

# Accept/reject thresholds (protocol §2 step 1). Applied to the NEW source's share.
INSTRUCTION_MAX = 0.10
SHA1_MAX = 0.05
REJECT_AT = 0.60

VERDICT_INDEPENDENT = "independent"
VERDICT_NEEDS_EXCLUSION = "needs_exclusion"
VERDICT_REJECT = "reject"

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")

# Near-duplicate detection. `MIN_INLIERS` was calibrated on EBench-18K vs I2EBench
# with a same-source control group: at >=50 RANSAC inliers the control produced
# 0 false positives in 210 comparisons, while known same-photo pairs scored 105-137.
MIN_INLIERS = 50
ORB_FEATURES = 500
NEAR_DUP_SAMPLE = 40  # originals drawn from the new source when --near-dup has no N

# ★ A RANSAC fit can be "verified" and still be meaningless, and the failure looks exactly
# like a real find. Measured 2026-08-01 while gating OmniEdit: 7 pairs cleared MIN_INLIERS
# (51-59) against unrelated LOW-TEXTURE incumbent images -- an AI-rendered caped woman vs a
# dark kitchen drawer, a fashion model vs a hazy cityscape. Those images yield only 10-267
# keypoints, so the Lowe ratio test degenerates, many source points match one target point,
# and `findHomography` fits a rank-deficient homography: |det H| = 0.000 and the inlier hull
# covers 0.0% of the other image in all seven. The tell is that the inlier count SCALES with
# the feature budget (51 -> 89 -> 148 at nfeatures 500/1000/2000) instead of saturating.
#
# ⚠️ A keypoint-count floor alone does NOT catch this -- 2 of the 7 cleared one. All three
# checks below are needed together.
#
# ⚠️ OFF BY DEFAULT so every published pool (the incumbent provenance cards) reproduces
# byte-for-byte. Turn it on for new work; a screened run reports both rates.
DEGENERATE_MIN_HULL_FRAC = 0.05
DEGENERATE_DET_BOUNDS = (0.01, 100.0)


def normalize_instruction(text: str) -> str:
    """lowercase -> drop punctuation -> collapse whitespace.

    Deliberately crude: two sources phrasing the same edit as "Add a hat." and
    "add a hat" must collide, and anything subtler than that belongs to a human
    reading the `examples` list in the report.
    """
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def file_sha1(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Sha1Cache:
    """Path -> sha1, invalidated by (size, mtime_ns). Optionally persisted.

    A source gets hashed once per run and re-hashed across runs only if the file
    actually changed, which matters because every `--against` comparison re-reads
    the same few thousand originals.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._data: Dict[str, str] = {}
        self._dirty = False
        if path is not None and path.is_file():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}  # a corrupt cache is a slow run, not a failure

    def get(self, path: Path) -> str:
        st = path.stat()
        key = f"{path.as_posix()}:{st.st_size}:{st.st_mtime_ns}"
        hit = self._data.get(key)
        if hit is None:
            hit = file_sha1(path)
            self._data[key] = hit
            self._dirty = True
        return hit

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")
        self._dirty = False


# --------------------------------------------------------------------------- #
# Near-duplicate dimension (opt-in; needs opencv)                              #
# --------------------------------------------------------------------------- #
def _cv2():
    """Lazy import — opencv is not a hard dependency of the pipeline."""
    try:
        import cv2  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "near-duplicate detection needs opencv-python (pip install opencv-python)"
        ) from exc
    return cv2


def orb_descriptors(path: Path, *, features: int = ORB_FEATURES):
    """Keypoints + descriptors at a normalised scale, or None if unreadable.

    Both sides are resized to a common longest side so a 512px crop and a 1024px
    original are compared in the same regime; ORB's own pyramid handles the rest.
    """
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > 640:
        scale = 640 / longest
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    kp, des = cv2.ORB_create(nfeatures=features).detectAndCompute(img, None)
    return None if des is None or len(kp) < 4 else (kp, des)


def _hull_area(points) -> float:
    """Convex-hull area of a point set; 0.0 for fewer than 3 points."""
    cv2 = _cv2()
    import numpy as np  # noqa: PLC0415 - only needed on this path

    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.float32(points).reshape(-1, 1, 2))))


def ransac_diagnostics(a, b) -> dict:
    """Everything `ransac_inliers` knows, so a suspicious match can be audited.

    `hull_frac_a` / `hull_frac_b` are the inlier hull as a share of the hull of ALL that
    side's keypoints — a self-contained proxy for "do the matches spread over the picture, or
    do they pile onto one spot?". A genuine same-photo match spreads; a degenerate fit does not.
    """
    cv2 = _cv2()
    import numpy as np  # noqa: PLC0415 - only needed on this path

    kp_a, des_a = a
    kp_b, des_b = b
    out = {"good": 0, "inliers": 0, "det_h": None,
           "hull_frac_a": 0.0, "hull_frac_b": 0.0,
           "keypoints_a": len(kp_a), "keypoints_b": len(kp_b)}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    out["good"] = len(good)
    if len(good) < 4:
        out["inliers"] = len(good)
        return out
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return out
    out["inliers"] = int(mask.sum())
    if h is not None:
        out["det_h"] = float(np.linalg.det(h))
    keep = mask.ravel().astype(bool)
    for side, kp, pts in (("a", kp_a, src), ("b", kp_b, dst)):
        whole = _hull_area([p.pt for p in kp])
        out[f"hull_frac_{side}"] = (
            _hull_area(pts[keep].reshape(-1, 2)) / whole if whole > 0 else 0.0
        )
    return out


def degenerate_reason(
    diag: dict,
    *,
    min_hull_frac: float = DEGENERATE_MIN_HULL_FRAC,
    det_bounds: Sequence[float] = DEGENERATE_DET_BOUNDS,
) -> Optional[str]:
    """Why this fit is not evidence of a shared photograph, or None if it is.

    See the DEGENERATE_* constants for the incident that produced these three checks, and
    why no one of them is sufficient alone.
    """
    if diag["inliers"] < 4:
        return None
    if diag["inliers"] > min(diag["keypoints_a"], diag["keypoints_b"]):
        # More inliers than one side has keypoints means many source points were matched
        # onto one target point — a pile-up, not a correspondence.
        return "more_inliers_than_keypoints"
    det = diag["det_h"]
    if det is None or not (det_bounds[0] <= abs(det) <= det_bounds[1]):
        return f"degenerate_homography(det={det})"
    if diag["hull_frac_a"] < min_hull_frac or diag["hull_frac_b"] < min_hull_frac:
        return (f"inliers_not_spread(a={diag['hull_frac_a']:.3f},"
                f"b={diag['hull_frac_b']:.3f})")
    return None


def ransac_inliers(a, b, *, screen_degenerate: bool = False,
                   min_hull_frac: float = DEGENERATE_MIN_HULL_FRAC,
                   det_bounds: Sequence[float] = DEGENERATE_DET_BOUNDS) -> int:
    """RANSAC-verified match count between two ORB signatures.

    Ratio-tested matches alone are not enough — repetitive texture produces plenty
    of them between unrelated photos. Requiring a consistent homography is what
    makes the threshold hold on a control group.

    `screen_degenerate=True` additionally returns **0** when the homography is degenerate
    (see the DEGENERATE_* constants). Off by default so published pools reproduce exactly.
    """
    diag = ransac_diagnostics(a, b)
    if screen_degenerate and degenerate_reason(
        diag, min_hull_frac=min_hull_frac, det_bounds=det_bounds
    ):
        return 0
    return diag["inliers"]


def near_duplicate_rate(
    new_originals: Sequence[str],
    other_originals: Sequence[str],
    *,
    root: Path,
    sample: int,
    seed: int = 42,
    min_inliers: int = MIN_INLIERS,
    signatures: Optional[dict] = None,
) -> dict:
    """Share of a seeded sample of `new_originals` that re-appears in `other_originals`.

    Sampled rather than exhaustive because the comparison is O(N*M) image matches;
    `sampled` is reported alongside the rate so the reader knows the precision of
    the estimate rather than mistaking it for a census.
    """
    signatures = signatures if signatures is not None else {}

    def sig(rel: str):
        if rel not in signatures:
            p = resolve(rel, root)
            signatures[rel] = orb_descriptors(p) if p.is_file() else None
        return signatures[rel]

    picked = sorted(new_originals)
    if sample < len(picked):
        picked = sorted(random.Random(f"{seed}:near_dup").sample(picked, sample))

    matches: List[dict] = []
    compared = 0
    for rel in picked:
        a = sig(rel)
        if a is None:
            continue
        compared += 1
        for other_rel in sorted(other_originals):
            b = sig(other_rel)
            if b is None:
                continue
            inliers = ransac_inliers(a, b)
            if inliers >= min_inliers:
                matches.append({"new": rel, "other": other_rel, "inliers": inliers})
                break  # one confirmed source is enough to call this original dirty
    return {
        "rate": _share(len(matches), compared),
        "sampled": compared,
        "min_inliers": min_inliers,
        "matches": matches[:10],
    }


@dataclass
class SourceIndex:
    """The two comparable fingerprints of one manifest, plus what went missing."""

    name: str
    path: Path
    n_records: int = 0
    instructions: Set[str] = field(default_factory=set)
    originals: Set[str] = field(default_factory=set)  # manifest-relative posix
    sha1s: Set[str] = field(default_factory=set)
    missing_files: int = 0

    @property
    def hashed(self) -> int:
        return len(self.originals) - self.missing_files


def _source_name(records: Sequence[SampleRecord], path: Path) -> str:
    """Prefer the records' own `source_dataset`; fall back to the file stem.

    A manifest holding several sources (e.g. the combined pool) has no single
    name, so the stem is used and the mixture stays visible in `n_records`.
    """
    names = Counter(r.source_dataset for r in records)
    if len(names) == 1:
        return next(iter(names))
    return path.stem


def build_index(
    manifest: PathLike,
    *,
    root: PathLike | None = None,
    with_pixels: bool = True,
    cache: Optional[Sha1Cache] = None,
    name: Optional[str] = None,
) -> SourceIndex:
    """Read a samples manifest into an overlap fingerprint.

    Missing image files are counted, never fatal: a source is routinely checked
    before its pixels are downloaded (protocol §2's `--dry-run` escape from the
    manifest-before-overlap chicken-and-egg).
    """
    root = Path(root) if root is not None else default_root()
    path = Path(manifest)
    records = io.read_jsonl(path, SampleRecord)
    idx = SourceIndex(name=name or _source_name(records, path), path=path,
                      n_records=len(records))
    for r in records:
        idx.instructions.add(normalize_instruction(r.instruction))
        idx.originals.add(r.original_image_path.as_posix())
    if not with_pixels:
        idx.missing_files = len(idx.originals)
        return idx

    cache = cache if cache is not None else Sha1Cache()
    for rel in sorted(idx.originals):
        p = resolve(rel, root)
        if not p.is_file():
            idx.missing_files += 1
            continue
        idx.sha1s.add(cache.get(p))
    return idx


def _share(matches: int, denom: int) -> Optional[float]:
    return round(matches / denom, 6) if denom else None


def _verdict(instruction: Optional[float], sha1: Optional[float],
             near_dup: Optional[float] = None) -> str:
    """Worst dimension decides. near-dup shares the pixel threshold — a cropped
    re-encode of someone else's photo is the same contamination as a byte copy."""
    dims = [v for v in (instruction, sha1, near_dup) if v is not None]
    if max(dims, default=0.0) >= REJECT_AT:
        return VERDICT_REJECT
    over_text = instruction is not None and instruction >= INSTRUCTION_MAX
    over_pixel = any(v is not None and v >= SHA1_MAX for v in (sha1, near_dup))
    return VERDICT_NEEDS_EXCLUSION if (over_text or over_pixel) else VERDICT_INDEPENDENT


def compare(new: SourceIndex, other: SourceIndex, *, n_examples: int = 5,
            near_dup: Optional[dict] = None) -> dict:
    """Overlap of `new` against `other`, both directions, with a verdict."""
    inst_hits = new.instructions & other.instructions
    sha_hits = new.sha1s & other.sha1s
    instruction_overlap = _share(len(inst_hits), len(new.instructions))
    sha1_overlap = _share(len(sha_hits), new.hashed) if new.sha1s or other.sha1s else None
    return {
        "instruction_overlap": instruction_overlap,
        "instruction_overlap_reverse": _share(len(inst_hits), len(other.instructions)),
        "instruction_matches": len(inst_hits),
        "sha1_overlap": sha1_overlap,
        "sha1_overlap_reverse": _share(len(sha_hits), other.hashed) if sha1_overlap is not None else None,
        "sha1_matches": len(sha_hits),
        "near_dup": near_dup,
        "n_records": other.n_records,
        "manifest": other.path.as_posix(),
        "verdict": _verdict(instruction_overlap, sha1_overlap,
                            near_dup["rate"] if near_dup else None),
        # Verbatim overlapping instructions so a human can tell a genuine shared
        # content pool from a coincidence like "remove the background".
        "examples": sorted(inst_hits)[:n_examples],
    }


def check(
    new_manifest: PathLike,
    against: Iterable[PathLike],
    *,
    root: PathLike | None = None,
    with_pixels: bool = True,
    cache_path: Optional[PathLike] = None,
    near_dup_sample: int = 0,
    seed: int = 42,
) -> dict:
    """Build the report documented in protocol §2 step 1.

    Top level holds one `vs_<source>` key per comparison (that shape is the
    published contract, referenced from `<src>.yaml`'s `provenance.overlap_report`);
    `_meta` carries everything about the run itself.
    """
    root = Path(root) if root is not None else default_root()
    cache = Sha1Cache(Path(cache_path)) if cache_path is not None else Sha1Cache()
    new = build_index(new_manifest, root=root, with_pixels=with_pixels, cache=cache)

    report: dict = {}
    verdicts: List[str] = []
    signatures: dict = {}  # ORB signatures reused across every `against` manifest
    for other_path in against:
        other = build_index(other_path, root=root, with_pixels=with_pixels, cache=cache)
        near = None
        if near_dup_sample:
            # Exact sha1 hits are already counted; near-dup answers the *rest*.
            near = near_duplicate_rate(
                sorted(new.originals), sorted(other.originals), root=root,
                sample=near_dup_sample, seed=seed, signatures=signatures,
            )
        key = f"vs_{other.name}"
        # Two manifests of the same source would collide; keep both, keyed by stem.
        if key in report:
            key = f"vs_{other.name}@{Path(other_path).stem}"
        report[key] = compare(new, other, near_dup=near)
        verdicts.append(report[key]["verdict"])
    cache.save()

    if VERDICT_REJECT in verdicts:
        overall = VERDICT_REJECT
    elif VERDICT_NEEDS_EXCLUSION in verdicts:
        overall = VERDICT_NEEDS_EXCLUSION
    else:
        overall = VERDICT_INDEPENDENT
    report["_meta"] = {
        "new_source": new.name,
        "new_manifest": Path(new_manifest).as_posix(),
        "new_records": new.n_records,
        "new_distinct_instructions": len(new.instructions),
        "new_distinct_originals": len(new.originals),
        "new_missing_files": new.missing_files,
        "pixels_compared": with_pixels,
        "near_dup_sample": near_dup_sample,
        "thresholds": {
            "instruction_max": INSTRUCTION_MAX,
            "sha1_max": SHA1_MAX,
            "reject_at": REJECT_AT,
            "min_inliers": MIN_INLIERS,
        },
        "verdict": overall,
    }
    return report


def exit_code_for(report: dict) -> int:
    """0 independent / 1 needs exclusion / 2 reject — usable as a CI gate."""
    return {VERDICT_INDEPENDENT: 0, VERDICT_NEEDS_EXCLUSION: 1, VERDICT_REJECT: 2}[
        report["_meta"]["verdict"]
    ]


def format_report(report: dict) -> str:
    meta = report["_meta"]
    lines = [
        f"new source: {meta['new_source']}  ({meta['new_records']} records, "
        f"{meta['new_distinct_instructions']} distinct instructions, "
        f"{meta['new_distinct_originals']} distinct originals)"
    ]
    if meta["new_missing_files"]:
        note = "pixel comparison skipped" if not meta["pixels_compared"] else "unhashable"
        lines.append(f"  ! {meta['new_missing_files']} original(s) {note}")
    lines.append(f"  {'against':<28}{'instr':>9}{'sha1':>9}{'neardup':>9}  verdict")
    for key, r in report.items():
        if key == "_meta":
            continue
        def pct(v):
            return "  n/a" if v is None else f"{v * 100:6.2f}%"
        near = r.get("near_dup")
        lines.append(f"  {key:<28}{pct(r['instruction_overlap']):>9}"
                     f"{pct(r['sha1_overlap']):>9}"
                     f"{pct(near['rate']) if near else '  n/a':>9}  {r['verdict']}")
        if r["examples"]:
            lines.append(f"      e.g. {r['examples'][0][:70]!r}")
        if near and near["matches"]:
            m = near["matches"][0]
            lines.append(f"      near-dup ({near['sampled']} sampled), e.g. "
                         f"{m['new']} ~ {m['other']} [{m['inliers']} inliers]")
    lines.append(f"  => {meta['verdict']}")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure content-pool overlap of a new source manifest (protocol R-B)."
    )
    p.add_argument("--new", type=Path, required=True, help="samples manifest of the new source")
    p.add_argument("--against", type=Path, nargs="+", required=True,
                   help="existing samples manifests to compare against")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--no-pixels", action="store_true",
                   help="text dimension only (use before images are downloaded)")
    p.add_argument("--near-dup", type=int, nargs="?", const=NEAR_DUP_SAMPLE, default=0,
                   metavar="N", help="also run ORB+RANSAC near-duplicate detection on a "
                                     f"seeded sample of N originals (default {NEAR_DUP_SAMPLE}); "
                                     "needed when the other source may have cropped/re-encoded")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", type=Path, default=Path("data/provenance/.sha1_cache.json"))
    p.add_argument("--dry-run", action="store_true", help="print what would be compared")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.dry_run:
        dims = "text only" if args.no_pixels else "text + pixels"
        if args.near_dup:
            dims += f" + near-dup (sample {args.near_dup})"
        print(f"[dry-run] {args.new} vs " + ", ".join(str(a) for a in args.against)
              + f" ({dims})")
        return 0
    report = check(args.new, args.against, root=args.root,
                   with_pixels=not args.no_pixels, cache_path=args.cache,
                   near_dup_sample=args.near_dup, seed=args.seed)
    print(format_report(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return exit_code_for(report)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
