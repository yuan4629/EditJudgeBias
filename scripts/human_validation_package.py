#!/usr/bin/env python
"""WP-A6a — the human quality-validation package for the six cues that have none.

    python scripts/human_validation_package.py build      # render + blind (free)
    python scripts/human_validation_package.py status      # how far the labelling got
    python scripts/human_validation_package.py unblind     # join labels back to cues

The first human package (150 labels, collected with an earlier script that is not part of
this repository) is kept as it was; this is a second, separate package.

★ TWO THINGS THIS DOES THAT THE FIRST PACKAGE DID NOT.

1. **It draws from the images the VALIDATORS actually judged.** The first package
   defaults to the pilot manifests while the validators ran the v2 manifest, and the two
   legs were only ever aligned by the `bias_type` string — so of 150 human-labelled
   images and 1,210 validator-judged images, exactly 5 are the same picture (3.3%).
   "Triple validation" then means three instruments on three different samples. Here the
   candidates are the intersection of the ids present and parsed in every
   `validation__*.jsonl`, so each human label lands on an image that already carries
   three validator verdicts, and human-vs-validator agreement becomes computable per
   image instead of per cue name.

2. **It hides the cue from the annotator.** This package includes `sham` — a visually
   null JPEG round-trip whose whole purpose is to measure the HUMAN false-flag floor.
   A floor measured by an annotator who can read `_sham` in the filename is not a floor;
   it is a memory test. So the 180 items are shuffled into one presentation order, named
   by index alone, and the index -> cue mapping is written to a separate key file that
   the annotator never opens. `unblind` joins it back afterwards.

   The rest of the protocol is deliberately IDENTICAL to the published 150 (No /
   Slightly / Yes on "did the modification change how well the editing task was
   completed", LEFT = the original edit, RIGHT = the modified one), because the two sets
   are reported in the same column and a protocol change would make them incomparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edit_judge_bias.data import io  # noqa: E402
from edit_judge_bias.data.manifest_utils import resolve  # noqa: E402
from edit_judge_bias.data.schema import BiasedRecord, SampleRecord  # noqa: E402
from validator_draw import published_draw  # noqa: E402  (one implementation of `_select`)

#: The six cues with `n_human == 0` in `quality_combined.csv`: the four C-class
#: region-touching injectors, the one B-class cue the first package missed, and the
#: placebo. `zoom_inset` is the contested one (the only cell in the study that fails the
#: 85% gate, on glm-4v) and `sham` is the floor that makes every other row readable.
CUES = (
    "region_annotation", "zoom_inset", "distraction",
    "detail_caption", "aesthetic_filter", "sham",
)

SAMPLES = REPO / "data" / "manifests" / "samples_judge_v2.jsonl"
BIASED = REPO / "data" / "manifests" / "biased_samples_full_v2.jsonl"
QUALITY = REPO / "results" / "v2" / "quality"
OUT_DIR = REPO / "data" / "human_validation_v2"
IAA_DIR = REPO / "data" / "human_validation_v2_iaa"
LABELS = "labels.csv"
KEY = "KEY_bias_assignment_do_not_open_while_labelling.csv"
RESOLVED = "human_validation_labels_v2.csv"
LABEL_COL = "quality_changed (No/Slightly/Yes)"

_CAPTION_H = 48
_GAP = 12
_MAX_H = 900


def validated_ids() -> Set[str]:
    """biased_ids that every validator judged AND parsed, within the PUBLISHED draw.

    Two intersections, each load-bearing.

    Across validators, not union: a human label is only worth its extra power if the
    same picture carries all three validator verdicts, and one validator's parse failure
    would otherwise leave a hole exactly where the comparison is made.

    With `_select(110, seed=42)`, because that is the draw every published quality number
    stands on — and because WP-A3 is APPENDING 670 new `sham` verdicts to these very
    files right now. Without this clamp the `sham` candidate pool would be 110 before the
    top-up and 780 after, so the same seed would produce a different package depending on
    what time of day it was run. A sample that is not reproducible cannot anchor a rate.
    """
    sets: List[Set[str]] = []
    for path in sorted(QUALITY.glob("validation__*.jsonl")):
        ok: Set[str] = set()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("parse_success") and r.get("biased_id"):
                    ok.add(r["biased_id"])
        sets.append(ok)
    if not sets:
        raise SystemExit(f"no validation__*.jsonl under {QUALITY}")
    draw = published_draw()
    published = {f"{base}__{cue}" for cue, bases in draw.items() for base in bases}
    return set.intersection(*sets) & published


def _load(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _composite(edited: Image.Image, biased: Image.Image, instruction: str) -> Image.Image:
    """Side by side at a COMMON scale, letterboxed — never stretched.

    Squashing 768x1344 frames into a square thumbnail makes an honest human answer
    impossible. Both members here are two versions
    of one edit, so any difference in apparent scale between the halves is an artefact of
    the renderer that the annotator would have to attribute to the cue. One scale factor,
    white padding for the shorter member.
    """
    h = max(edited.height, biased.height)
    scale = min(1.0, _MAX_H / h)
    def fit(im):
        if scale < 1.0:
            return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        return im
    a, b = fit(edited), fit(biased)
    ph = max(a.height, b.height)
    w = a.width + _GAP + b.width
    canvas = Image.new("RGB", (w, ph + _CAPTION_H), (255, 255, 255))
    canvas.paste(a, (0, _CAPTION_H + (ph - a.height) // 2))
    canvas.paste(b, (a.width + _GAP, _CAPTION_H + (ph - b.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"Instruction: {instruction[:90]}", fill=(0, 0, 0))
    draw.text((4, 24), "LEFT = Edit A (original)      RIGHT = Edit B (modified)", fill=(0, 0, 0))
    return canvas


def cmd_build(args) -> int:
    eligible = validated_ids()
    by_id = {s.sample_id: s for s in io.read_jsonl(SAMPLES, SampleRecord)}
    by_cue: Dict[str, List[BiasedRecord]] = defaultdict(list)
    for b in io.read_jsonl(BIASED, BiasedRecord):
        if b.bias_type in CUES and b.biased_id in eligible:
            by_cue[b.bias_type].append(b)

    rng = random.Random(args.seed)
    picked: List[BiasedRecord] = []
    for cue in CUES:
        bucket = sorted(by_cue.get(cue, []), key=lambda b: b.biased_id)
        print(f"  {cue:20} validated candidates={len(bucket):4}  taking {args.per_cue}")
        if len(bucket) < args.per_cue:
            print(f"    REFUSING: only {len(bucket)} candidates for {cue}")
            return 1
        rng.shuffle(bucket)
        picked.extend(bucket[: args.per_cue])

    # One presentation order across all six cues. This is what blinds the package: an
    # annotator who meets `sham` in a run of thirty consecutive placebos has been told
    # the answer, whatever the filenames say.
    rng.shuffle(picked)

    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels, key = [], []
    for i, b in enumerate(picked, start=1):
        base = by_id.get(b.base_sample_id)
        if base is None:
            print(f"    REFUSING: {b.biased_id} has no base sample in {SAMPLES.name}")
            return 1
        fname = f"{i:03d}.png"
        _composite(
            _load(resolve(base.edited_image_path, REPO)),
            _load(resolve(b.biased_image_path, REPO)),
            base.instruction,
        ).save(img_dir / fname)
        labels.append({"index": i, "image_file": f"images/{fname}",
                       "instruction": base.instruction, LABEL_COL: ""})
        key.append({"index": i, "image_file": f"images/{fname}", "bias_type": b.bias_type,
                    "biased_id": b.biased_id, "base_sample_id": b.base_sample_id})

    _write_csv(out / LABELS, labels)
    _write_csv(out / KEY, key)
    (out / "INSTRUCTIONS.txt").write_text(_INSTRUCTIONS, encoding="utf-8")
    print(f"wrote {len(labels)} items -> {out}")
    print(f"  label this : {out / LABELS}")
    print(f"  do NOT open: {out / KEY}  (it names the cue for every row)")
    return 0


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_status(args) -> int:
    out = Path(args.out_dir)
    rows = _read_csv(out / LABELS)
    done = [r for r in rows if (r.get(LABEL_COL) or "").strip()]
    bad = sorted({(r.get(LABEL_COL) or "").strip() for r in done} - {"No", "Slightly", "Yes"})
    print(f"labelled {len(done)}/{len(rows)}")
    if bad:
        print(f"  ⚠ unrecognised labels (must be exactly No/Slightly/Yes): {bad}")
    return 0


def cmd_unblind(args) -> int:
    """Join the labels back onto the cues, in the column names the table builder reads."""
    out = Path(args.out_dir)
    labels = {int(r["index"]): (r.get(LABEL_COL) or "").strip() for r in _read_csv(out / LABELS)}
    key = _read_csv(out / KEY)
    rows, skipped = [], 0
    for k in key:
        lab = labels.get(int(k["index"]), "")
        if lab not in ("No", "Slightly", "Yes"):
            skipped += 1
            continue
        rows.append({"index": k["index"], "image_file": k["image_file"],
                     "bias_type": k["bias_type"], "biased_id": k["biased_id"],
                     "base_sample_id": k["base_sample_id"], LABEL_COL: lab})
    if not rows:
        print("nothing labelled yet — run `status` and fill in labels.csv first")
        return 1
    _write_csv(out / RESOLVED, rows)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["bias_type"]][r[LABEL_COL]] += 1
    print(f"wrote {out / RESOLVED}  rows={len(rows)}  unlabelled skipped={skipped}")
    for cue in CUES:
        c = counts.get(cue, Counter())
        n = sum(c.values())
        print(f"  {cue:20} n={n:3}  No={c['No']:3} Slightly={c['Slightly']:3} Yes={c['Yes']:3}")
    return 0


def cmd_build_iaa(args) -> int:
    """WP-A6c: a second annotator's subset, for Cohen's kappa.

    THREE PROPERTIES, EACH OF WHICH KAPPA IS MEANINGLESS WITHOUT.

    1. **The images are COPIED, never re-rendered.**  Agreement is only agreement if both
       annotators looked at the same picture.  Re-running `_composite` would produce the
       same picture today and an unknown picture after any edit to the compositor, and the
       failure would be invisible: kappa would quietly become a measure of how much the
       renderer drifted.  So this reads the primary package's KEY file and copies bytes.
    2. **It is renumbered and re-shuffled.**  Sharing the primary package's indices would
       let one annotator's remark ("47 was odd") land on the other's row 47, and would let
       the two sheets be compared before both are finished.  The join back is by
       `biased_id`, carried in this package's own key file.
    3. **It is stratified by cue at the same 1-in-6 rate**, so the placebo share is
       identical (10 of 60, as 30 of 180) and kappa is not dominated by one cue.  A
       kappa computed on a subset whose cue mix differs from the primary package's is not
       an agreement rate for the primary package.

    The seed differs from the primary build's on purpose: `--per-cue 10` with the same
    seed would hand the second annotator a prefix of the first annotator's own draw
    order, which is a needless correlation between the two sheets.
    """
    src = Path(args.out_dir)
    dst = Path(args.iaa_dir)
    key = _read_csv(src / KEY)
    if not key:
        print(f"REFUSING: no key file at {src / KEY} — build the primary package first")
        return 1

    by_cue: Dict[str, List[dict]] = defaultdict(list)
    for row in key:
        by_cue[row["bias_type"]].append(row)

    rng = random.Random(args.seed)
    picked: List[dict] = []
    for cue in CUES:
        bucket = sorted(by_cue.get(cue, []), key=lambda r: r["biased_id"])
        print(f"  {cue:20} in primary package={len(bucket):3}  taking {args.per_cue}")
        if len(bucket) < args.per_cue:
            print(f"    REFUSING: only {len(bucket)} rows for {cue}")
            return 1
        rng.shuffle(bucket)
        picked.extend(bucket[: args.per_cue])
    rng.shuffle(picked)

    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels, newkey = [], []
    src_labels = {int(r["index"]): r for r in _read_csv(src / LABELS)}
    for i, row in enumerate(picked, start=1):
        fname = f"{i:03d}.png"
        src_png = src / row["image_file"]
        if not src_png.exists():
            print(f"    REFUSING: {src_png} is missing")
            return 1
        shutil.copyfile(src_png, img_dir / fname)
        instruction = src_labels[int(row["index"])]["instruction"]
        labels.append({"index": i, "image_file": f"images/{fname}",
                       "instruction": instruction, LABEL_COL: ""})
        newkey.append({"index": i, "image_file": f"images/{fname}",
                       "bias_type": row["bias_type"], "biased_id": row["biased_id"],
                       "base_sample_id": row["base_sample_id"],
                       "primary_index": row["index"]})

    _write_csv(dst / LABELS, labels)
    _write_csv(dst / KEY, newkey)
    (dst / "INSTRUCTIONS.txt").write_text(_INSTRUCTIONS_IAA, encoding="utf-8")
    print(f"wrote {len(labels)} items -> {dst}")
    print(f"  hand to annotator 2 : {dst / LABELS}  (+ images/, + INSTRUCTIONS.txt)")
    print(f"  do NOT hand over    : {dst / KEY}  (names the cue AND the primary index)")
    return 0


def cmd_kappa(args) -> int:
    """Cohen's kappa between the two annotators on the shared subset."""
    prim = {r["biased_id"]: r[LABEL_COL] for r in _read_csv(Path(args.out_dir) / RESOLVED)}
    iaa_key = {int(r["index"]): r for r in _read_csv(Path(args.iaa_dir) / KEY)}
    iaa: Dict[str, str] = {}
    for r in _read_csv(Path(args.iaa_dir) / LABELS):
        lab = (r.get(LABEL_COL) or "").strip()
        if lab in ("No", "Slightly", "Yes"):
            iaa[iaa_key[int(r["index"])]["biased_id"]] = lab

    shared = sorted(set(prim) & set(iaa))
    if not shared:
        print("no overlap yet — annotator 1 needs `unblind`, annotator 2 needs labels")
        return 1
    missing = sorted(set(iaa) - set(prim))
    if missing:
        print(f"⚠️  {len(missing)} rows labelled by annotator 2 only; kappa uses the {len(shared)} shared")

    cats = ("No", "Slightly", "Yes")
    obs = sum(prim[b] == iaa[b] for b in shared) / len(shared)
    exp = sum(
        (sum(prim[b] == c for b in shared) / len(shared))
        * (sum(iaa[b] == c for b in shared) / len(shared))
        for c in cats
    )
    kappa = (obs - exp) / (1 - exp) if exp < 1 else float("nan")
    # The binary collapse is the one the paper's pass/fail gate actually uses:
    # No+Slightly = preserved, Yes = not preserved.  Report both -- a three-level kappa
    # can look poor purely from No/Slightly boundary disagreement that the gate ignores.
    binp = {b: (prim[b] == "Yes") for b in shared}
    bini = {b: (iaa[b] == "Yes") for b in shared}
    obs_b = sum(binp[b] == bini[b] for b in shared) / len(shared)
    py, iy = sum(binp.values()) / len(shared), sum(bini.values()) / len(shared)
    exp_b = py * iy + (1 - py) * (1 - iy)
    kappa_b = (obs_b - exp_b) / (1 - exp_b) if exp_b < 1 else float("nan")

    out = {"n_shared": len(shared), "observed_agreement": round(obs, 4),
           "expected_agreement": round(exp, 4), "cohens_kappa_3level": round(kappa, 4),
           "observed_agreement_binary": round(obs_b, 4),
           "cohens_kappa_binary_yes_vs_rest": round(kappa_b, 4),
           "annotator1_yes_rate": round(py, 4), "annotator2_yes_rate": round(iy, 4)}
    path = Path(args.iaa_dir) / "iaa_kappa.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"  {k:34} {v}")
    print(f"wrote {path}")
    return 0


_INSTRUCTIONS = """WP-A6a human quality-validation package (180 items)

WHAT TO DO
  For each row of labels.csv, open images/<image_file>.
  LEFT  = Edit A: the original edited image.
  RIGHT = Edit B: the same edit after a cosmetic modification.

THE ONE QUESTION
  Did the modification change how well the EDITING TASK was completed?
  Answer exactly one of:  No  /  Slightly  /  Yes
  Only No and Slightly count as quality-preserved.

WHAT NOT TO JUDGE
  Ignore that B may simply look different -- brighter, annotated, captioned, with an
  extra sticker or an inset. The question is never "is B modified" (it always is); it is
  whether the underlying EDIT RESULT changed. If the instruction was "remove the car"
  and the car is still gone in B, the answer is No even if B is covered in overlays.

WHY THE FILES ARE NUMBERED AND NOT NAMED
  The cue is hidden on purpose. Some of these 180 items are a placebo -- a re-encode
  that changes nothing visible -- and they exist to measure how often a modification is
  reported where there is none. That number is only meaningful if you cannot tell which
  rows they are. KEY_bias_assignment_do_not_open_while_labelling.csv holds the mapping;
  please do not open it until every row is labelled.

IF YOU ARE UNSURE
  Use Slightly. It is a real category, not a hedge: it means "changed, but not enough to
  change which of two edits I would prefer".
"""

_INSTRUCTIONS_IAA = _INSTRUCTIONS.replace(
    "WP-A6a human quality-validation package (180 items)",
    "WP-A6c human quality-validation package -- SECOND ANNOTATOR (60 items)",
).replace(
    "Some of these 180 items are a placebo",
    "Some of these 60 items are a placebo",
) + """
PLEASE DO NOT DISCUSS THE ITEMS WITH THE OTHER ANNOTATOR UNTIL BOTH SHEETS ARE DONE
  These 60 items are a subset of a 180-item sheet another person is labelling, and the
  point of this sheet is to measure how far two people independently agree. Comparing
  notes first does not improve that number, it deletes it. The numbering here is
  deliberately different from the other sheet's, so "row 12" means nothing in common.
"""


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render the blinded package")
    b.add_argument("--per-cue", type=int, default=30)
    b.add_argument("--seed", type=int, default=123)
    b.set_defaults(fn=cmd_build)
    sub.add_parser("status", help="how many rows are labelled").set_defaults(fn=cmd_status)
    sub.add_parser("unblind", help="join labels to cues after labelling").set_defaults(fn=cmd_unblind)
    ap.add_argument("--iaa-dir", default=str(IAA_DIR))
    i = sub.add_parser("build-iaa", help="WP-A6c: 60-item subset for a second annotator")
    i.add_argument("--per-cue", type=int, default=10)
    i.add_argument("--seed", type=int, default=777)  # NOT 123: see cmd_build_iaa
    i.set_defaults(fn=cmd_build_iaa)
    sub.add_parser("kappa", help="Cohen's kappa once both sheets are done").set_defaults(fn=cmd_kappa)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
