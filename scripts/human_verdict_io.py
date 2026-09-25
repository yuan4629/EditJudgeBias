"""Round-trip the human validation sheets through a CSV a person can open in Excel.

Hand-editing JSONL is how a verdict silently lands on the wrong row: the files carry
82 / 31 / 20 lines of dense JSON and one misplaced quote breaks the whole file. This
script exports the pending rows to a CSV with one row per image, and imports the
filled CSV back, writing *only* the ``human_verdict`` field and leaving every other
key byte-identical.

    python scripts/human_verdict_io.py export dg_pairs -o verdicts_dg.csv
    # ... open verdicts_dg.csv, fill the `verdict` column, save as CSV ...
    python scripts/human_verdict_io.py import dg_pairs -i verdicts_dg.csv

Import refuses unknown labels rather than writing them, because a typo that reaches
the JSONL is counted in the denominator but never in the numerator by
``metrics/construct_validity.py`` — i.e. it silently lowers the pass rate.

Both directions are idempotent and resumable: export skips rows already filled
(``--all`` to include them), and import only rewrites rows whose verdict changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from edit_judge_bias.metrics.construct_validity import SOURCES  # noqa: E402

# Which field holds the image, and which fields are worth showing the annotator while
# they judge. Kept here rather than on ValidationSource: these are presentation
# concerns for the CSV, not part of the metric contract.
_IMAGE_FIELD = {"dg_pairs": "image", "ds_v3": "sheet", "ds_v4": "sheet"}
_CONTEXT_FIELDS = {
    # `auditors_agree` first: the 40 rows where it is False are the ones the D-G arm
    # exists to resolve (the two MLLM auditors agree at chance, kappa = -0.028).
    "dg_pairs": ["auditors_agree", "attribute", "left", "right"],
    "ds_v3": ["instruction", "person_frac", "skin_frac"],
    "ds_v4": ["instruction", "person_frac", "skin_frac"],
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def cmd_export(source_name: str, out: Path, include_done: bool) -> int:
    src = SOURCES[source_name]
    vocab = src.vocabulary
    image_field = _IMAGE_FIELD[source_name]
    context = _CONTEXT_FIELDS[source_name]

    rows = _load(REPO_ROOT / src.path)
    pending = [r for r in rows if include_done or not str(r.get("human_verdict", "")).strip()]

    if source_name == "dg_pairs":
        # Disagreement rows first. The two MLLM auditors agree at chance overall
        # (kappa = -0.028), so the 40 rows where they disagree are the ones a human
        # is actually needed for -- and an annotator who runs out of time should have
        # spent it on those, not on the 42 the auditors already agree about.
        pending.sort(key=lambda r: (bool(r.get("auditors_agree", True)), r[src.key_field]))

    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# source={source_name}   question: {vocab.question}"])
        w.writerow([f"# allowed verdicts: {' | '.join(vocab.labels)}"])
        w.writerow([f"# counts as PASS: {', '.join(vocab.pass_labels)}"])
        w.writerow(["key", "image", "verdict", "note"] + context)
        for r in pending:
            w.writerow(
                [r[src.key_field], r.get(image_field, ""), r.get("human_verdict", ""), ""]
                + [r.get(c, "") for c in context]
            )

    print(f"exported {len(pending)} of {len(rows)} rows -> {out}")
    print(f"  question:         {vocab.question}")
    print(f"  allowed verdicts: {', '.join(vocab.labels)}")
    if not pending:
        print("  (nothing pending; pass --all to re-export filled rows)")
    return 0


def cmd_import(source_name: str, inp: Path, dry_run: bool) -> int:
    src = SOURCES[source_name]
    jsonl = REPO_ROOT / src.path
    rows = _load(jsonl)
    by_key = {r[src.key_field]: r for r in rows}

    filled: dict[str, str] = {}
    bad: list[tuple[str, str]] = []
    unknown_keys: list[str] = []

    with inp.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(line for line in fh if not line.startswith("#")):
            key = (rec.get("key") or "").strip()
            verdict = (rec.get("verdict") or "").strip().lower()
            if not key or not verdict:
                continue
            if key not in by_key:
                unknown_keys.append(key)
            elif verdict not in src.vocabulary.labels:
                bad.append((key, verdict))
            else:
                filled[key] = verdict

    if bad or unknown_keys:
        for key, verdict in bad:
            print(
                f"  REJECTED  {key}: '{verdict}' is not one of {src.vocabulary.labels}",
                file=sys.stderr,
            )
        for key in unknown_keys:
            print(f"  REJECTED  unknown key not in {src.path}: {key}", file=sys.stderr)
        # ASCII only: this goes to a Windows console that is often GBK, where a
        # non-ASCII dash renders as mojibake right where the error must be readable.
        print(
            f"\n{len(bad) + len(unknown_keys)} bad row(s) -- nothing was written. "
            "Fix the CSV and re-run.",
            file=sys.stderr,
        )
        return 1

    changed = sum(1 for k, v in filled.items() if by_key[k].get("human_verdict", "") != v)
    if dry_run:
        print(f"[dry-run] would write {changed} changed verdict(s) of {len(filled)} filled -> {src.path}")
        return 0

    if changed == 0:
        # Never rewrite on a no-op. Re-serialising JSONL is not guaranteed to reproduce
        # the original byte-for-byte, and these sheets are hand-annotated data that
        # nothing else can regenerate — a pointless rewrite is pure downside.
        print(f"no change ({len(filled)} verdict(s) already match) — {src.path} left untouched")
        return 0

    for key, verdict in filled.items():
        by_key[key]["human_verdict"] = verdict
    _dump(jsonl, rows)

    done = sum(1 for r in rows if str(r.get("human_verdict", "")).strip())
    print(f"wrote {changed} changed verdict(s) -> {src.path}")
    print(f"  annotated: {done}/{len(rows)}")
    print("  next: bash scripts/construct_validity.sh")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="write pending rows to a CSV")
    e.add_argument("source", choices=sorted(SOURCES))
    e.add_argument("-o", "--out", type=Path, required=True)
    e.add_argument("--all", action="store_true", help="include rows that already have a verdict")

    i = sub.add_parser("import", help="read a filled CSV back into the JSONL")
    i.add_argument("source", choices=sorted(SOURCES))
    i.add_argument("-i", "--in", dest="inp", type=Path, required=True)
    i.add_argument("--dry-run", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "export":
        return cmd_export(a.source, a.out, a.all)
    return cmd_import(a.source, a.inp, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
