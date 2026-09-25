"""Freeze the human-adjudication verdicts into `construct_validity.csv`.

    bash scripts/construct_validity.sh                       # the wrapper
    python -m edit_judge_bias.experiments.build_construct_validity_table --dry-run
    python -m edit_judge_bias.experiments.build_construct_validity_table

Zero API. Reads three staged adjudication sheets READ-ONLY (`data/human_validation_fairness/
pairs.jsonl`, `data/human_validation_ds/index.jsonl`, `data/human_validation_ds_v4/
index.jsonl`) and writes one frozen table. It never writes a verdict — filling those files
in is the human's job and this module is only their consumer.

★ IT IS SAFE AND CHEAP TO RE-RUN, WHICH IS THIS PIPELINE'S FORM OF "RESUMABLE". There is no
checkpoint to keep because there is no expensive step: the whole computation is a census of
133 rows. Run it again after every batch of verdicts you fill in; the CSV is rewritten from
the sheets each time, so a half-filled sheet yields a half-filled table stamped
`status=partial` with `n_annotated` beside `n_total` rather than a stale one.

★ ONE FILE FOR TWO ARMS, ON PURPOSE. D-G's sheets belong to the `results/v2_fairness` tree
and D-S's to `results/v2_fairness_ds`, but they answer the same question — *at what rate do
the constructed stimuli actually instantiate the construct?* — so they are frozen into one
table under the D-S tree with an `arm` column. Splitting them would invite a reader to quote
one arm's rate for the other; they are pooled in a FILE, never in a NUMBER (there is no
combined row).

★ WHAT THE TABLE MUST NOT BE USED FOR. `pass_rate` is a property of the *stimulus corpus*,
not of a judge and not of a result. It gates what may be claimed:
  * D-S: a majority of `patch`/`weak` means the arm reports `sham_nonskin` as a control and
    makes NO fairness claim (README_GATE_P4.md) -- a publishable negative result about the
    corpus, not a failed run.
  * D-G: the arm was already stopped at the gate; these verdicts decide, on the record,
    which of the two disagreeing MLLM auditors that stop was right to follow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.experiments.build_claim_tables import write_csv
from edit_judge_bias.metrics.construct_validity import (
    READING_RULE,
    SOURCES,
    evaluate_all,
)

DEFAULT_OUT = "results/v2_fairness_ds/metrics/construct_validity.csv"


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build(
    *,
    root: Optional[Path] = None,
    sources: Optional[Sequence[str]] = None,
    out: str = DEFAULT_OUT,
    dry_run: bool = False,
) -> dict:
    root = Path(root) if root is not None else default_root()
    summaries = evaluate_all(root=root, sources=sources)
    if not summaries:
        return {
            "rows": 0,
            "error": (
                "no adjudication sheets found on disk — expected at least one of: "
                + ", ".join(s.path for s in SOURCES.values())
            ),
        }

    rows = [s.as_row() for s in summaries]
    overall = [s for s in summaries if s.stratum == "all"]
    n_total = sum(s.n_total for s in overall)
    n_annotated = sum(s.n_annotated for s in overall)

    report = {
        "rows": len(rows),
        "sources": [s.source for s in overall],
        "n_total": n_total,
        "n_annotated": n_annotated,
        "messages": [s.message() for s in overall],
        "reading": READING_RULE,
        "out": out,
        "dry_run": bool(dry_run),
    }
    if n_annotated == 0:
        # Not an error — it is the current, expected state, and it is stated rather than
        # rendered as a rate of 0.0. The table is still written so the empty shape (and its
        # n_total denominators) is on disk and reviewable.
        report["note"] = (
            f"0 of {n_total} sheets annotated: NO pass rate exists yet. "
            "Fill `human_verdict` in the sheets listed above, then re-run this command."
        )
    if not dry_run:
        path = root / out
        write_csv(path, rows)
        report["out"] = str(path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Freeze human construct-validity verdicts into a CSV (zero API)."
    )
    ap.add_argument("--config", default=None, help="optional YAML overriding any flag below")
    ap.add_argument("--root", default=None)
    ap.add_argument(
        "--sources",
        default=None,
        help=f"comma-separated subset of {sorted(SOURCES)} (default: all staged)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report annotation coverage without writing the CSV",
    )
    a = ap.parse_args(argv)

    cfg = load_config(Path(a.config)) if a.config else {}
    root = cfg.get("root", a.root)
    out = cfg.get("out", a.out)
    raw_sources = cfg.get("sources", a.sources)
    sources: Optional[List[str]]
    if raw_sources is None:
        sources = None
    elif isinstance(raw_sources, str):
        sources = [s.strip() for s in raw_sources.split(",") if s.strip()]
    else:
        sources = list(raw_sources)

    rep = build(
        root=Path(root) if root else None,
        sources=sources,
        out=out,
        dry_run=bool(a.dry_run or cfg.get("dry_run", False)),
    )
    for msg in rep.get("messages", []):
        print(msg)
    if "note" in rep:
        print("\n" + rep["note"])
    if "error" in rep:
        print("ERROR: " + rep["error"])
        return 1
    print("\nreading rule: " + rep["reading"])
    print(f"\n{rep['rows']} rows -> {rep['out']}" + ("  (dry run, not written)" if rep["dry_run"] else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
