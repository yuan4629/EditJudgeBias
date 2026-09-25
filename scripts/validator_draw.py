#!/usr/bin/env python
"""The validators' matched `sham` floor (WP-A3) and the C-class ablation draw (WP-A4c).

The published validator runs drew 110 images per cue. Only a few of those base images
also had a `sham` verdict, so a per-image (McNemar) comparison of a cue with `sham` was
not possible. This arm adds `sham` verdicts for every base image drawn for any cue: 670
new rows, a union of 737. The three validator manifests are only ever appended to, and
the runner resumes by `biased_id`, so the risk is not destroying data but paying for
2,010 calls that leave the design unmatched. The acceptance check therefore ships with
the setup.

    python scripts/validator_draw.py manifest   # step 1  (free)  -> the 670-row manifest
    python scripts/validator_draw.py backup     # before paying   -> *.presham copies
    ...  run_quality_validation x3 with the shammatch configs   # step 3 (paid)
    python scripts/validator_draw.py verify     # acceptance criteria 1-3
    python scripts/validator_draw.py restore    # the inverse of the paid run

Every subcommand is idempotent and prints what it did.

WHY `manifest` CROSS-CHECKS AGAINST DISK BEFORE IT WRITES ANYTHING.
The design rests on reproducing `run_quality_validation._select(110, seed=42)` exactly:
the same un-reset RNG walk that produced the small overlap in the first place. If the
reproduction drifted by one shuffle, the 670 ids would be a different draw and every cue
would come back partially unmatched. So the reproduction is checked against the
`base_sample_id` sets present in the three `validation__*.jsonl` files, and the command
refuses to write on a mismatch. The C-class re-render (WP-A2) rewrote 273 rows of the main
manifest in place without changing any `biased_id`; this check proves that instead of
assuming it.

WHY THE PAIRING CHECK IS A COMMAND RATHER THAN A COMMENT.
"670 new sham rows landed" and "the design is now matched" are different claims. The
second one is criterion 3, the only one that makes the McNemar columns meaningful.
`verify` asserts it per cue and per validator, and prints each cue's coverage, so a
partial run shows up as a number instead of as a silently unmatched table.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
MAIN_MANIFEST = REPO / "data" / "manifests" / "biased_samples_full_v2.jsonl"
SHAMMATCH_MANIFEST = REPO / "data" / "manifests" / "biased_samples_shammatch_v2.jsonl"
QUALITY = REPO / "results" / "v2" / "quality"
VALIDATION_GLOB = "validation__*.jsonl"
BACKUP_SUFFIX = ".presham"

#: `configs/experiment/quality_validation_full_v2*.yaml` — the published draw.
PER_BIAS = 110
SEED = 42
CONTROL_BIAS = "sham"

#: Measured: the union of the 10 cues' base images, and how many of them still
#: need a sham verdict. Asserted rather than merely printed — if either number moves,
#: the published draw moved, and nothing downstream should be trusted.
EXPECT_UNION = 737
EXPECT_NEED = 670


# --------------------------------------------------------------------------- #
# the published draw                                                          #
# --------------------------------------------------------------------------- #
def _manifest_rows() -> List[Tuple[str, str, str, str]]:
    """(biased_id, base_sample_id, bias_type, raw_line) for the full v2 manifest."""
    out: List[Tuple[str, str, str, str]] = []
    with MAIN_MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append((r["biased_id"], r["base_sample_id"], r["bias_type"], line))
    return out


def published_draw() -> Dict[str, Set[str]]:
    """bias_type -> the 110 `base_sample_id`s that were actually validated.

    A line-by-line reproduction of `run_quality_validation._select`: ONE `random.Random`
    constructed outside the loop (`:64`), buckets sorted by `biased_id` (`:67`) and
    shuffled without a seed reset (`:68`), prefix of `per_bias` taken (`:69`). The
    missing reset is the defect this arm exists to repair, so it must be reproduced, not
    corrected.
    """
    by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for biased_id, base_id, bias_type, _ in _manifest_rows():
        by_type[bias_type].append((biased_id, base_id))
    rng = random.Random(SEED)
    draw: Dict[str, Set[str]] = {}
    for bt in sorted(by_type):
        bucket = sorted(by_type[bt], key=lambda pair: pair[0])
        rng.shuffle(bucket)
        draw[bt] = {base_id for _, base_id in bucket[:PER_BIAS]}
    return draw


def _validated_base_ids(path: Path) -> Dict[str, Set[str]]:
    """bias_type -> base_sample_ids present in one `validation__*.jsonl`."""
    out: Dict[str, Set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            bid = r.get("biased_id") or ""
            if "__" not in bid:
                continue
            base, _, bias = bid.rpartition("__")
            out[bias].add(base)
    return out


def _validator_files() -> List[Path]:
    return sorted(QUALITY.glob(VALIDATION_GLOB))


def _cross_check(draw: Dict[str, Set[str]]) -> List[str]:
    """Compare the reproduced draw against every validator file on disk."""
    problems: List[str] = []
    for path in _validator_files():
        on_disk = _validated_base_ids(path)
        for bias, ids in sorted(on_disk.items()):
            want = draw.get(bias, set())
            # A validator may be mid-run and hold a subset; it may never hold an id the
            # draw did not select. Extra ids mean the reproduction is wrong.
            extra = ids - want
            if extra:
                problems.append(
                    f"{path.name}: {bias} has {len(extra)} id(s) outside the reproduced "
                    f"draw, e.g. {sorted(extra)[:3]}"
                )
            elif len(ids) != len(want):
                problems.append(
                    f"{path.name}: {bias} has {len(ids)}/{len(want)} of the draw "
                    "(incomplete run, not a reproduction failure)"
                )
    return problems


# --------------------------------------------------------------------------- #
# subcommands                                                                 #
# --------------------------------------------------------------------------- #
def cmd_manifest(_args) -> int:
    draw = published_draw()
    problems = _cross_check(draw)
    hard = [p for p in problems if "outside the reproduced draw" in p]
    for p in problems:
        print(("  FAIL " if p in hard else "  note ") + p)
    if hard:
        print("\nREFUSING to write: the reproduced draw does not match what is on disk.")
        return 1

    arms = {bt: ids for bt, ids in draw.items() if bt != CONTROL_BIAS}
    union: Set[str] = set().union(*arms.values())
    need = sorted(union - draw[CONTROL_BIAS])
    print(f"arms={len(arms)}  union base ids={len(union)}  sham rows to validate={len(need)}")
    if len(union) != EXPECT_UNION or len(need) != EXPECT_NEED:
        print(f"REFUSING to write: expected {EXPECT_UNION}/{EXPECT_NEED} (the published draw).")
        return 1

    covered = draw[CONTROL_BIAS] | set(need)
    for bt, ids in sorted(arms.items()):
        assert len(ids & covered) == PER_BIAS, f"{bt}: only {len(ids & covered)}/{PER_BIAS} matched"
    print(f"all {len(arms)} arms fully matched once these {len(need)} land: OK")

    by_biased_id = {bid: line for bid, _, _, line in _manifest_rows()}
    missing = [s for s in need if f"{s}__{CONTROL_BIAS}" not in by_biased_id]
    if missing:
        print(f"REFUSING to write: {len(missing)} sham row(s) absent from the manifest, "
              f"e.g. {missing[:3]}")
        return 1
    SHAMMATCH_MANIFEST.write_text(
        "".join(by_biased_id[f"{s}__{CONTROL_BIAS}"] for s in need), encoding="utf-8"
    )
    print(f"wrote {SHAMMATCH_MANIFEST.relative_to(REPO)}  rows={len(need)}")
    return 0


#: WP-A4c — the four cues that touch or overlay the edit region.
C_CLASS = ("region_annotation", "zoom_inset", "detail_caption", "distraction")
ABLATION_MANIFEST = REPO / "data" / "manifests" / "biased_samples_cclass_ablation.jsonl"


def cmd_ablation_manifest(_args) -> int:
    """Write the C-class cues' PUBLISHED images for the WP-A4c prompt ablation.

    WHY IT MUST BE THE PUBLISHED IMAGES AND NOT A FRESH DRAW.  The ablation asks one
    question: does naming C-class overlays as "cosmetic" in the validator template change
    the verdict?  Answering it needs the same pictures, the same validator, and only the
    prompt different -- then a difference is attributable to the prompt.  A fresh draw
    would reintroduce exactly the confound WP-A3 just spent $36 removing ("or these
    pictures were harder"), one level up.

    So this reuses `published_draw()`, the same function that reproduced the published
    validator sampling byte-for-byte before A3 was allowed to spend anything.
    """
    draw = published_draw()
    missing = [c for c in C_CLASS if c not in draw]
    if missing:
        print(f"published draw has no rows for {missing}", file=sys.stderr)
        return 1

    wanted = {(base, cue) for cue in C_CLASS for base in draw[cue]}
    kept: List[str] = []
    for _biased_id, base, bias_type, line in _manifest_rows():
        if (base, bias_type) in wanted:
            kept.append(line.rstrip("\n"))

    if len(kept) != len(wanted):
        print(f"expected {len(wanted)} rows, matched {len(kept)} — refusing to write",
              file=sys.stderr)
        return 1
    expect = PER_BIAS * len(C_CLASS)
    if len(kept) != expect:
        print(f"expected {expect} rows ({PER_BIAS} x {len(C_CLASS)} cues), got {len(kept)}",
              file=sys.stderr)
        return 1

    ABLATION_MANIFEST.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"wrote {len(kept)} rows ({', '.join(C_CLASS)}) -> {ABLATION_MANIFEST}")
    return 0


def cmd_backup(_args) -> int:
    n = 0
    for path in _validator_files():
        dest = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        if dest.exists():
            print(f"  keep  {dest.name} (already exists — not overwriting a backup)")
            continue
        shutil.copy2(path, dest)
        print(f"  saved {dest.name}  ({sum(1 for _ in path.open(encoding='utf-8'))} rows)")
        n += 1
    print(f"backed up {n} file(s)")
    return 0


def cmd_restore(_args) -> int:
    n = 0
    for path in _validator_files():
        src = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        if not src.exists():
            print(f"  skip  {path.name} (no {BACKUP_SUFFIX} copy)")
            continue
        shutil.copy2(src, path)
        print(f"  restored {path.name}")
        n += 1
    print(f"restored {n} file(s)")
    return 0


def cmd_verify(_args) -> int:
    ok = True
    draw = published_draw()
    arms = {bt: ids for bt, ids in draw.items() if bt != CONTROL_BIAS}

    # criterion 1 — the manifest itself
    if not SHAMMATCH_MANIFEST.exists():
        print("  FAIL manifest missing — run `manifest` first")
        return 1
    rows = [json.loads(l) for l in SHAMMATCH_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = [r["biased_id"] for r in rows if r["bias_type"] != CONTROL_BIAS]
    print(f"  [1] manifest rows={len(rows)} (want {EXPECT_NEED})  non-sham rows={len(bad)} (want 0)")
    ok &= len(rows) == EXPECT_NEED and not bad

    for path in _validator_files():
        seen: Dict[str, int] = defaultdict(int)
        parse_fail_new = 0
        total = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                seen[r.get("biased_id") or ""] += 1
                bid = r.get("biased_id") or ""
                base = bid.rpartition("__")[0]
                if bid.endswith(f"__{CONTROL_BIAS}") and base not in draw[CONTROL_BIAS]:
                    if not r.get("parse_success", False):
                        parse_fail_new += 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        # criterion 2 — row count, no duplicates, new rows parsed
        print(f"  [2] {path.name}: rows={total} (want {PER_BIAS * 11 + EXPECT_NEED}) "
              f"duplicates={len(dupes)} new-row parse failures={parse_fail_new}")
        ok &= not dupes and parse_fail_new == 0

        # criterion 3 — the definitional check: is the design actually matched?
        on_disk = _validated_base_ids(path)
        sham_ids = on_disk.get(CONTROL_BIAS, set())
        worst = None
        for bt, ids in sorted(arms.items()):
            cov = len(ids & sham_ids)
            if worst is None or cov < worst[1]:
                worst = (bt, cov)
            if cov != PER_BIAS:
                print(f"      {bt}: {cov}/{PER_BIAS} of this arm's base images have a sham verdict")
                ok = False
        print(f"  [3] {path.name}: worst arm coverage {worst[1]}/{PER_BIAS} ({worst[0]})")

    print("VERIFY: " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (
        ("manifest", cmd_manifest, "reproduce the published draw and write the 670-row manifest"),
        ("backup", cmd_backup, "copy each validation__*.jsonl to *.presham"),
        ("verify", cmd_verify, "acceptance criteria 1-3 (manifest, row counts, matched design)"),
        ("restore", cmd_restore, "put the *.presham copies back"),
        ("ablation-manifest", cmd_ablation_manifest,
         "WP-A4c: the four C-class cues' PUBLISHED images, for the prompt ablation"),
    ):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
