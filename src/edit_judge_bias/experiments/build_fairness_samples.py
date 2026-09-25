"""Turn step-4 renders into `SampleRecord`s so the existing scoring runner can judge them.

The fairness arm needs no new judge code at all: a D-class item is just an
(original, instruction, edited) triple like any other, where the "original" is the
counterfactual original and the "edited" is the step-4 render. Everything the paired metric
needs later rides in `SampleMetadata` (`extra="allow"`), which is the same mechanism
`subset_block` / `anchor_source` already use and which `matches_filter` can already read.

★ THE PRESERVATION GATE IS APPLIED HERE, AND IT IS APPLIED PAIRWISE.
Only renders whose (base_sample_id, attribute) pair passed `run_attribute_validation` are
emitted, and a pair is admitted only if BOTH its variants survive. Keeping a lone variant
would leave an orphan that the paired metric silently drops later — the drop belongs here,
where it is counted and reported, not in the metric where it would be invisible.

Excluding the gate entirely (`--no-gate`) exists for smoke tests only; the study numbers must
be gated, because an ungated pair may be one scene twice (the flip failed) or two different
scenes (the flip changed more than the attribute), and those bias the gap in opposite
directions.

    python -m edit_judge_bias.experiments.build_fairness_samples --report
    python -m edit_judge_bias.experiments.build_fairness_samples
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.data.schema import ContentCategory, SampleRecord
from edit_judge_bias.experiments.run_attribute_validation import passed_pair_keys
from edit_judge_bias.fairness.records import CounterfactualEditRecord


#: Auditor manifests that gate the study set. They combine by INTERSECTION, so listing one
#: that has not been collected yet is harmless — only files present on disk gate.
DEFAULT_VALIDATIONS = (
    "results/v2_fairness/quality/attribute_validation__gpt-4o-mini.jsonl",
    "results/v2_fairness/quality/attribute_validation__gpt-5.5.jsonl",
)


def build(
    *,
    root: Optional[Path] = None,
    pool: str = "data/manifests/samples_fairness_v2.jsonl",
    renders: str = "data/manifests/counterfactual_edits.jsonl",
    validation: Optional[Sequence[str]] = DEFAULT_VALIDATIONS,
    out_path: str = "data/manifests/samples_fairness_judge_v2.jsonl",
    write: bool = True,
) -> dict:
    root = Path(root) if root is not None else default_root()
    base_by_id = {s.sample_id: s for s in io.read_jsonl(root / pool, SampleRecord)}
    renders_path = root / renders
    if not renders_path.exists():
        # A legitimate "step 4 has not run yet" state, not a bug — say so instead of raising a
        # traceback out of the JSONL reader.
        return {
            "renders_in": 0,
            "samples_out": 0,
            "error": f"step-4 manifest not found: {renders}. "
                     "Run experiments.run_counterfactual_edit first.",
            "out_path": None,
        }
    rows = io.read_jsonl(renders_path, CounterfactualEditRecord)

    # Every auditor manifest present on disk gates, by intersection (see passed_pair_keys).
    gate: Optional[set] = None
    gate_sources: List[str] = []
    if validation:
        names = [validation] if isinstance(validation, str) else list(validation)
        present = [n for n in names if (root / n).exists()]
        if present:
            gate = passed_pair_keys([root / n for n in present])
            gate_sources = present

    # Group by (base, attribute, render_index) so the pairwise admission can be checked before
    # anything is emitted.
    groups: Dict[tuple, List[CounterfactualEditRecord]] = collections.defaultdict(list)
    for r in rows:
        groups[(r.base_sample_id, r.attribute, r.render_index)].append(r)

    records: List[SampleRecord] = []
    dropped = collections.Counter()
    for (base, attr, ridx), members in sorted(groups.items()):
        if base not in base_by_id:
            dropped["base_sample_missing_from_pool"] += len(members)
            continue
        if gate is not None and f"{base}__{attr}" not in gate:
            dropped["failed_preservation_gate"] += len(members)
            continue
        # render_index 1 is the study render and must be a complete two-variant pair. The
        # null-control renders (index >= 2) are deliberately single-variant, so they are exempt.
        if ridx == 1 and len(members) != 2:
            dropped["incomplete_variant_pair"] += len(members)
            continue
        base_rec = base_by_id[base]
        for r in members:
            rec = SampleRecord(
                sample_id=r.render_id,
                source_dataset=base_rec.source_dataset,
                edit_type=base_rec.edit_type,
                content_category=ContentCategory.HUMAN,
                original_image_path=r.counterfactual_image_path,
                instruction=r.instruction,
                edit_model=r.editor_model,
                edited_image_path=r.edited_image_path,
            )
            rec.metadata.subset_block = "fairness"
            rec.metadata.base_sample_id = base
            rec.metadata.attribute = attr
            rec.metadata.variant_label = r.variant_label
            rec.metadata.render_index = r.render_index
            rec.metadata.is_null_control = r.render_index > 1
            records.append(rec)

    report = {
        "renders_in": len(rows),
        "samples_out": len(records),
        "gate_applied": gate is not None,
        "gate_passed_pairs": (len(gate) if gate is not None else None),
        "gate_sources": gate_sources,
        "dropped": dict(dropped),
        "by_attribute": dict(
            sorted(collections.Counter(r.metadata.attribute for r in records).items())
        ),
        "study_renders": sum(1 for r in records if not r.metadata.is_null_control),
        "null_control_renders": sum(1 for r in records if r.metadata.is_null_control),
        "out_path": out_path if write else None,
    }
    if write:
        io.write_jsonl(root / out_path, records)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Emit SampleRecords for the D-class scoring arm.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--pool", default="data/manifests/samples_fairness_v2.jsonl")
    ap.add_argument("--renders", default="data/manifests/counterfactual_edits.jsonl")
    ap.add_argument(
        "--validation", action="append", default=None,
        help="auditor manifest; repeat for several (they gate by INTERSECTION)",
    )
    ap.add_argument("--no-gate", action="store_true", help="smoke tests only — see the docstring")
    ap.add_argument("--out", default="data/manifests/samples_fairness_judge_v2.jsonl")
    ap.add_argument("--report", action="store_true", help="print counts, write nothing")
    a = ap.parse_args(argv)
    rep = build(
        root=Path(a.root) if a.root else None,
        pool=a.pool,
        renders=a.renders,
        validation=(None if a.no_gate else (a.validation or DEFAULT_VALIDATIONS)),
        out_path=a.out,
        write=not a.report,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
