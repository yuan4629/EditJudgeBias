#!/usr/bin/env python
"""Check a local rebuild of the benchmark against the published fingerprints.

    python scripts/verify_benchmark.py upstream    # the downloaded corpus metadata files
    python scripts/verify_benchmark.py manifests   # row sets of every manifest (seconds)
    python scripts/verify_benchmark.py sources     # bytes of the judged source images
    python scripts/verify_benchmark.py images      # pixels of every rendered cue image (minutes)
    python scripts/verify_benchmark.py all

The fingerprint file (`data/provenance/benchmark_v2_fingerprints.json`) holds counts and
SHA-256 digests only -- no rows, ids, text or pixels -- so it ships with the code while
the data stays with its upstream owners. Digests are grouped (by source for samples and
pairs, by cue for rendered images) so that a mismatch names the group that differs.

Digest rules (all SHA-256, hex):
  upstream   over the bytes of each upstream metadata file the builders read (instruction
             and label files, rating tables, parquet shards), one digest per file.
  manifests  over the file's non-empty lines, each stripped of its line ending, sorted and
             joined with "\\n" -- so a manifest written in a different row order still matches.
  sources    over sorted "<path>\\t<sha256 of the file bytes>" lines for every original and
             edited image referenced by the judged samples and the pair members, grouped as
             "<source>/original" and "<source>/<editor>". The builders store upstream bytes
             verbatim, so these are the upstream files themselves.
  images     over sorted "<id>\\t<sha256 of the decoded pixels>" lines, where the pixel digest
             covers "<mode>|<width>x<height>|" followed by the raw pixel buffer. Decoded pixels,
             not file bytes, so the check does not depend on the PNG encoder or zlib version.

Exit status is 0 when every checked group matches and 1 otherwise. Missing files are
reported as missing rather than as mismatches. Maintainers regenerate the file from a
known-good tree with `--write`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FINGERPRINTS = Path("data/provenance/benchmark_v2_fingerprints.json")
FORMAT = 1

M = "data/manifests/"


def _by_field(field: str) -> Callable[[dict], str]:
    return lambda row: str(row.get(field))


def _by_id_prefix(field: str) -> Callable[[dict], str]:
    return lambda row: str(row.get(field, "")).split("_", 1)[0]


#: manifest -> how its rows are grouped. Samples by source, pairs by the id prefix of their
#: first member (i2e / ih / gb / ebench), rendered-image manifests by cue.
MANIFESTS: Dict[str, Callable[[dict], str]] = {
    # per-source builds
    M + "samples_i2e600.jsonl": _by_field("source_dataset"),
    M + "pairs_i2e600.jsonl": _by_id_prefix("sample_id_a"),
    M + "samples_imagenhub.jsonl": _by_field("source_dataset"),
    M + "pairs_imagenhub.jsonl": _by_id_prefix("sample_id_a"),
    M + "samples_genaibench.jsonl": _by_field("source_dataset"),
    M + "pairs_genaibench.jsonl": _by_id_prefix("sample_id_a"),
    M + "samples_magicbrush.jsonl": _by_field("source_dataset"),
    M + "samples_ebench18k.jsonl": _by_field("source_dataset"),
    M + "pairs_ebench18k.jsonl": _by_id_prefix("sample_id_a"),
    # combined pool and judged subset
    M + "samples_full_v2.jsonl": _by_field("source_dataset"),
    M + "pairs_full_v2.jsonl": _by_id_prefix("sample_id_a"),
    M + "samples_judge_v2.jsonl": _by_field("source_dataset"),
    M + "pairs_judge_v2.jsonl": _by_id_prefix("sample_id_a"),
    M + "samples_pair_members_v2.jsonl": _by_field("source_dataset"),
    # rendered cues and the validator draws built on them
    M + "biased_samples_full_v2.jsonl": _by_field("bias_type"),
    M + "biased_pair_members_v2.jsonl": _by_field("bias_type"),
    M + "biased_pair_members_fill_v2.jsonl": _by_field("bias_type"),
    M + "samples_editdamage_v2.jsonl": _by_field("source_dataset"),
    M + "biased_samples_editdamage_v2.jsonl": _by_field("bias_type"),
    M + "biased_samples_editdamage_v2_100.jsonl": _by_field("bias_type"),
    M + "biased_samples_shammatch_v2.jsonl": _by_field("bias_type"),
    M + "biased_samples_cclass_ablation.jsonl": _by_field("bias_type"),
    M + "quality_pair_members_fill_v2_p4.jsonl": _by_field("bias_type"),
}

#: rendered-image manifests whose images are fingerprinted by pixels.
IMAGE_MANIFESTS = (
    M + "biased_samples_full_v2.jsonl",
    M + "biased_pair_members_v2.jsonl",
    M + "biased_pair_members_fill_v2.jsonl",
    M + "biased_samples_editdamage_v2.jsonl",
)

#: upstream files (relative to the root) whose bytes are fingerprinted one by one.
UPSTREAM_GLOBS = (
    "tmp_data/EditBench/EditData/*/*.json",
    "tmp_data/imagenhub_meta/dataset_lookup.json",
    "tmp_data/imagenhub_meta/dataset_lookup.csv",
    "tmp_data/imagenhub_meta/rater*.tsv",
    "tmp_data/imagenhub_museum/tree_main.json",
    "tmp_data/ebench_meta/*.json",
    "tmp_data/hf_cache/datasets--TIGER-Lab--GenAI-Bench/snapshots/*/image_edition/test-00000-of-00001.parquet",
    "tmp_data/hf_cache/datasets--osunlp--MagicBrush/snapshots/*/data/dev-*.parquet",
)

#: sample manifests whose original + edited images are fingerprinted by file bytes.
SOURCE_MANIFESTS = (
    M + "samples_judge_v2.jsonl",
    M + "samples_pair_members_v2.jsonl",
)


# --------------------------------------------------------------------------- #
# digests                                                                      #
# --------------------------------------------------------------------------- #
def _sha(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(lines).encode("utf-8"))
    return h.hexdigest()


def _lines(path: Path) -> List[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [ln.rstrip("\r\n") for ln in fh if ln.strip()]


def manifest_fingerprint(path: Path, group_of: Callable[[dict], str]) -> dict:
    lines = _lines(path)
    groups: Dict[str, List[str]] = defaultdict(list)
    for ln in lines:
        groups[group_of(json.loads(ln))].append(ln)
    return {
        "rows": len(lines),
        "sha256": _sha(sorted(lines)),
        "groups": {g: {"rows": len(v), "sha256": _sha(sorted(v))}
                   for g, v in sorted(groups.items())},
    }


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_sha(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        h = hashlib.sha256(f"{im.mode}|{im.width}x{im.height}|".encode("ascii"))
        h.update(im.tobytes())
    return h.hexdigest()


def _hash_many(fn: Callable[[Path], str], paths: List[Path], workers: int) -> List[Optional[str]]:
    def one(p: Path) -> Optional[str]:
        return fn(p) if p.is_file() else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, paths))


def image_fingerprint(root: Path, manifest: Path, workers: int) -> Tuple[dict, int]:
    rows = [json.loads(ln) for ln in _lines(manifest)]
    ids = [r["biased_id"] for r in rows]
    shas = _hash_many(_pixel_sha, [root / r["biased_image_path"] for r in rows], workers)
    groups: Dict[str, List[str]] = defaultdict(list)
    missing = 0
    for r, i, s in zip(rows, ids, shas):
        if s is None:
            missing += 1
            s = "MISSING"
        groups[r["bias_type"]].append(f"{i}\t{s}")
    return ({g: {"images": len(v), "sha256": _sha(sorted(v))}
             for g, v in sorted(groups.items())}, missing)


def source_fingerprint(root: Path, manifests: Iterable[Path], workers: int) -> Tuple[dict, int]:
    by_path: Dict[str, str] = {}
    for manifest in manifests:
        for ln in _lines(manifest):
            r = json.loads(ln)
            src = r["source_dataset"]
            if r.get("original_image_path"):
                by_path[r["original_image_path"]] = f"{src}/original"
            if r.get("edited_image_path"):
                by_path[r["edited_image_path"]] = f"{src}/{r.get('edit_model')}"
    paths = sorted(by_path)
    shas = _hash_many(_file_sha, [root / p for p in paths], workers)
    groups: Dict[str, List[str]] = defaultdict(list)
    missing = 0
    for p, s in zip(paths, shas):
        if s is None:
            missing += 1
            s = "MISSING"
        groups[by_path[p]].append(f"{p}\t{s}")
    return ({g: {"files": len(v), "sha256": _sha(sorted(v))}
             for g, v in sorted(groups.items())}, missing)


# --------------------------------------------------------------------------- #
# build / compare                                                              #
# --------------------------------------------------------------------------- #
def _environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for mod in ("PIL", "numpy", "scipy", "skimage", "cv2"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional modules
            env[mod] = None
    try:
        from PIL import features

        env["freetype2"] = features.version("freetype2")
        env["libjpeg_turbo"] = features.version("libjpeg_turbo")
        env["zlib"] = features.version("zlib")
    except Exception:  # noqa: BLE001
        pass
    return env


def upstream_fingerprint(root: Path) -> dict:
    files = sorted({p for pattern in UPSTREAM_GLOBS for p in root.glob(pattern) if p.is_file()
                    # I2EBench: only <Category>/<Category>.json is read, not the _2nd.. rounds
                    and not ("EditBench" in p.parts and p.stem != p.parent.name)})
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": _file_sha(p)}
            for p in files}


def build(root: Path, parts: List[str], workers: int) -> dict:
    out: dict = {}
    if "upstream" in parts:
        out["upstream"] = upstream_fingerprint(root)
    if "manifests" in parts:
        out["manifests"] = {rel: manifest_fingerprint(root / rel, g)
                            for rel, g in MANIFESTS.items() if (root / rel).is_file()}
    if "sources" in parts:
        groups, missing = source_fingerprint(root, [root / m for m in SOURCE_MANIFESTS], workers)
        if missing:
            raise SystemExit(f"--write refused: {missing} source images are missing")
        out["sources"] = {"manifests": list(SOURCE_MANIFESTS), "groups": groups}
    if "images" in parts:
        out["images"] = {}
        for rel in IMAGE_MANIFESTS:
            groups, missing = image_fingerprint(root, root / rel, workers)
            if missing:
                raise SystemExit(f"--write refused: {missing} images of {rel} are missing")
            out["images"][rel] = groups
    return out


class Report:
    def __init__(self) -> None:
        self.ok = self.bad = self.missing = 0

    def line(self, status: str, what: str, detail: str = "") -> None:
        if status == "OK":
            self.ok += 1
        elif status == "MISSING":
            self.missing += 1
        else:
            self.bad += 1
        print(f"  [{status:<8}] {what}" + (f"  ({detail})" if detail else ""))


def _compare_groups(rep: Report, label: str, want: dict, have: dict, unit: str) -> None:
    for g in sorted(set(want) | set(have)):
        w, h = want.get(g), have.get(g)
        if w is None:
            rep.line("EXTRA", f"{label} :: {g}", f"{h[unit]} {unit} not in the fingerprints")
        elif h is None:
            rep.line("MISSING", f"{label} :: {g}", f"expected {w[unit]} {unit}")
        elif w == h:
            rep.line("OK", f"{label} :: {g}", f"{h[unit]} {unit}")
        else:
            rep.line("DIFFERS", f"{label} :: {g}", f"{h[unit]} {unit}, expected {w[unit]}")


def verify(root: Path, ref: dict, parts: List[str], workers: int) -> Report:
    rep = Report()
    if "upstream" in parts:
        print("upstream (downloaded corpus metadata)")
        for rel, want in sorted(ref.get("upstream", {}).items()):
            path = root / rel
            if not path.is_file():
                rep.line("MISSING", rel)
            elif _file_sha(path) == want["sha256"]:
                rep.line("OK", rel)
            else:
                rep.line("DIFFERS", rel, f"{path.stat().st_size} bytes, expected {want['bytes']}")
    if "manifests" in parts:
        print("manifests")
        for rel, want in ref.get("manifests", {}).items():
            path = root / rel
            if not path.is_file():
                rep.line("MISSING", rel)
                continue
            have = manifest_fingerprint(path, MANIFESTS[rel])
            if have["sha256"] == want["sha256"]:
                rep.line("OK", rel, f"{have['rows']} rows")
            else:
                _compare_groups(rep, rel, want["groups"], have["groups"], "rows")
    if "sources" in parts and "sources" in ref:
        print("sources (file bytes of the judged originals and edits)")
        manifests = [root / m for m in ref["sources"]["manifests"]]
        absent = [m for m in manifests if not m.is_file()]
        if absent:
            for m in absent:
                rep.line("MISSING", m.relative_to(root).as_posix(), "build the manifests first")
        else:
            have, missing = source_fingerprint(root, manifests, workers)
            if missing:
                print(f"  note: {missing} source files are missing")
            _compare_groups(rep, "sources", ref["sources"]["groups"], have, "files")
    if "images" in parts:
        print("images (decoded pixels of the rendered cues)")
        for rel, want in ref.get("images", {}).items():
            path = root / rel
            if not path.is_file():
                rep.line("MISSING", rel, "render the cues first")
                continue
            have, missing = image_fingerprint(root, path, workers)
            if missing:
                print(f"  note: {missing} images of {rel} are missing")
            _compare_groups(rep, rel, want, have, "images")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("upstream", "manifests", "sources", "images", "all"))
    ap.add_argument("--root", type=Path, default=REPO, help="project root (default: this repo)")
    ap.add_argument("--fingerprints", type=Path, default=None,
                    help=f"fingerprint file (default: <root>/{DEFAULT_FINGERPRINTS.as_posix()})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write", action="store_true",
                    help="maintainers: (re)write the checked parts from this tree")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    fp_path = args.fingerprints or (REPO / DEFAULT_FINGERPRINTS)
    parts = (["upstream", "manifests", "sources", "images"] if args.what == "all"
             else [args.what])

    if args.write:
        data = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.is_file() else {}
        data.update({"format": FORMAT, "benchmark": "EditJudgeBias v2", "seed": 42})
        data.update(build(root, parts, args.workers))
        data["environment"] = _environment()
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
        print(f"wrote {fp_path}")
        return 0

    ref = json.loads(fp_path.read_text(encoding="utf-8"))
    if ref.get("format") != FORMAT:
        print(f"unsupported fingerprint format {ref.get('format')!r}", file=sys.stderr)
        return 2
    rep = verify(root, ref, parts, args.workers)
    print(f"\n{rep.ok} match, {rep.bad} differ, {rep.missing} missing")
    if rep.bad:
        print("See docs/REPRODUCTION.md ('When a group differs') for the usual causes.")
    return 0 if rep.bad == 0 and rep.missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
