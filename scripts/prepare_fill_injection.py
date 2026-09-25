#!/usr/bin/env python
"""FILL v2 -- the free, local steps before any judge is asked, and their acceptance.

    PYTHONPATH=src python scripts/prepare_fill_injection.py snapshot
    PYTHONPATH=src python scripts/prepare_fill_injection.py seed
    PYTHONPATH=src python -m edit_judge_bias.experiments.run_bias_injection \
        --config configs/experiment/bias_injection_pair_members_fill_v2.yaml
    PYTHONPATH=src python scripts/prepare_fill_injection.py audit

snapshot  Records (size, mtime_ns) of every file under data/images/biased/ -- the tree
          this fill must not write to. Refuses to overwrite an existing snapshot, because
          it has to be the BEFORE state (`--force` if you can say why).
seed      Writes the records of the pair members that are also scoring samples into the
          fill manifest, copied VERBATIM from biased_samples_full_v2.jsonl, so
          run_bias_injection resumes past them and those members keep their published
          pixels. Idempotent.
audit     Acceptance for the injection. Exits non-zero if any hard check fails; writes
          results/logs/fill_v2_injection_audit.json either way.

Why each step exists is written in configs/experiment/bias_injection_pair_members_fill_v2.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

COND = ("saturation", "watermark", "aesthetic_filter", "region_annotation",
        "zoom_inset", "detail_caption", "distraction", "sham")
MEMBERS = REPO / "data" / "manifests" / "samples_pair_members_v2.jsonl"
FULL = REPO / "data" / "manifests" / "biased_samples_full_v2.jsonl"
OUT = REPO / "data" / "manifests" / "biased_pair_members_fill_v2.jsonl"
PROTECTED = REPO / "data" / "images" / "biased"
NEW_DIR = "data/images/biased_pair_fill"
SNAPSHOT = REPO / "results" / "logs" / "fill_v2_protected_images_snapshot.json"
AUDIT_OUT = REPO / "results" / "logs" / "fill_v2_injection_audit.json"
FAIL_LOG = REPO / "results" / "logs" / "bias_injection_failures_pair_members_fill_v2.jsonl"


def _jsonl(path: Path) -> Iterator[Tuple[str, dict]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n"), json.loads(line)


def _members() -> List[str]:
    return [r["sample_id"] for _, r in _jsonl(MEMBERS)]


def _full() -> Dict[str, Tuple[str, dict]]:
    return {r["biased_id"]: (line, r) for line, r in _jsonl(FULL)}


def _walk(root: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(REPO).as_posix()] = [st.st_size, st.st_mtime_ns]
    return out


# --------------------------------------------------------------------------- #
def cmd_snapshot(args) -> int:
    if SNAPSHOT.exists() and not args.force:
        print(f"REFUSING: {SNAPSHOT.relative_to(REPO)} exists. It must be the state BEFORE "
              "injection; pass --force only if no fill injection has run since it was taken.")
        return 1
    files = _walk(PROTECTED)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"root": "data/images/biased", "n_files": len(files),
                                    "files": files}, sort_keys=True), encoding="utf-8")
    print(f"snapshot: {len(files):,} files under data/images/biased -> "
          f"{SNAPSHOT.relative_to(REPO).as_posix()}")
    return 0


def _expected_seed() -> Tuple[int, List[str]]:
    full = _full()
    members = _members()
    shared = 0
    for sid in members:
        have = [f"{sid}__{c}" in full for c in COND]
        if any(have) and not all(have):
            raise SystemExit(f"{sid}: only some of the eight conditions are in the scoring "
                             "manifest -- the shared/new split is not clean, stop and look")
        shared += all(have)
    lines = [full[f"{sid}__{c}"][0] for c in COND for sid in members if f"{sid}__{c}" in full]
    return shared, lines


def cmd_seed(_args) -> int:
    shared, lines = _expected_seed()
    for line in lines:
        row = json.loads(line)
        if not (REPO / row["biased_image_path"]).is_file():
            raise SystemExit(f"seed record points at a missing image: {row['biased_image_path']}")
    if OUT.exists():
        have = [ln for ln, _ in _jsonl(OUT)]
        if have[:len(lines)] == lines:
            print(f"seed: already seeded ({len(lines)} records), nothing to do")
            return 0
        print(f"REFUSING: {OUT.relative_to(REPO).as_posix()} exists and does not begin with "
              "the expected seed records")
        return 1
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"seed: {shared} shared members x {len(COND)} = {len(lines)} records -> "
          f"{OUT.relative_to(REPO).as_posix()}")
    return 0


def cmd_audit(_args) -> int:
    from PIL import Image, ImageChops

    from edit_judge_bias.experiments.build_sensitivity_tables import CHANGED_PIXEL_THRESHOLD

    members = _members()
    full = _full()
    rows = [r for _, r in _jsonl(OUT)] if OUT.exists() else []
    checks: Dict[str, dict] = {}

    ids = [r["biased_id"] for r in rows]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    expected = {f"{sid}__{c}" for sid in members for c in COND}
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    checks["complete_and_unique"] = {
        "ok": not dup and not missing and not extra, "rows": len(rows),
        "expected": len(expected), "duplicates": len(dup), "missing": len(missing),
        "extra": len(extra), "missing_examples": missing[:5],
    }

    shared_n = shared_same = new_n = new_ok = 0
    bad: List[str] = []
    for r in rows:
        if r["biased_id"] in full:
            shared_n += 1
            shared_same += r == full[r["biased_id"]][1]
        else:
            new_n += 1
            p = r["biased_image_path"]
            good = p.startswith(f"{NEW_DIR}/{r['bias_type']}/") and (REPO / p).is_file()
            new_ok += good
            if not good:
                bad.append(p)
    checks["shared_records_verbatim"] = {"ok": shared_same == shared_n, "n": shared_n,
                                         "identical": shared_same}
    checks["new_images_only_in_fill_dir"] = {"ok": new_ok == new_n, "n": new_n,
                                             "bad_examples": bad[:5]}

    if not SNAPSHOT.exists():
        checks["protected_tree_untouched"] = {"ok": False,
                                              "reason": "no snapshot -- run `snapshot` first"}
    else:
        snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["files"]
        now = _walk(PROTECTED)
        changed = [p for p, v in snap.items() if now.get(p) != v]
        added = [p for p in now if p not in snap]
        checks["protected_tree_untouched"] = {
            "ok": not changed and not added, "files": len(snap),
            "changed_or_missing": len(changed), "added": len(added),
            "examples": (changed + added)[:5],
        }

    n_fail = sum(1 for _ in _jsonl(FAIL_LOG)) if FAIL_LOG.exists() else 0
    checks["no_injection_failures"] = {"ok": n_fail == 0, "failures": n_fail}

    # ---- informational: does the new pool look like the published one? ----
    region = {}
    for cue in ("region_annotation", "zoom_inset", "distraction"):
        region[cue] = {
            "fill_new": dict(Counter(r["bias_params"].get("region_method") for r in rows
                                     if r["bias_type"] == cue and r["biased_id"] not in full)),
            "published_scoring": dict(Counter(r["bias_params"].get("region_method")
                                              for _, r in full.values() if r["bias_type"] == cue)),
        }
    edited = {r["sample_id"]: r["edited_image_path"] for _, r in _jsonl(MEMBERS)}
    inside = total = 0
    inside_ids: List[str] = []
    for r in rows:
        if r["bias_type"] != "distraction" or r["biased_id"] in full:
            continue
        total += 1
        box = r["bias_params"].get("avoided_bbox")
        a = Image.open(REPO / edited[r["base_sample_id"]]).convert("RGB")
        b = Image.open(REPO / r["biased_image_path"]).convert("RGB")
        mask = ImageChops.difference(a, b).convert("L").point(
            lambda p: 255 if p > CHANGED_PIXEL_THRESHOLD else 0)
        if box and mask.crop(tuple(box)).getbbox() is not None:
            inside += 1
            inside_ids.append(r["biased_id"])
    info = {
        "region_method": region,
        "distraction_ink_inside_estimated_edit_region": {
            "n_new": total, "inside": inside, "examples": inside_ids[:10],
            "reads": "sticker pixels (edited vs biased, per-channel > threshold) inside the "
                     "region the injector was told to avoid; unavoidable when no slot fits",
        },
    }

    ok = all(c["ok"] for c in checks.values())
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps({"ok": ok, "checks": checks, "info": info},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, c in checks.items():
        print(f"  [{'OK ' if c['ok'] else 'BAD'}] {name}: "
              + ", ".join(f"{k}={v}" for k, v in c.items() if k != "ok"))
    d = info["distraction_ink_inside_estimated_edit_region"]
    print(f"  info: distraction ink inside the estimated edit region {d['inside']}/{d['n_new']}")
    for cue, v in region.items():
        print(f"  info: {cue} region_method new={v['fill_new']} published={v['published_scoring']}")
    print(f"audit {'PASSED' if ok else 'FAILED'} -> {AUDIT_OUT.relative_to(REPO).as_posix()}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--force", action="store_true")
    sub.add_parser("seed")
    sub.add_parser("audit")
    args = ap.parse_args(argv)
    return {"snapshot": cmd_snapshot, "seed": cmd_seed, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
