"""Render figures from the metric CSVs (Milestone 5 + the paper figure set).

Reads whatever tables are present, so it works on the pilot archive (score shift +
position only), on the v2 grid (which also has the frozen claim tables), and on the
D-S fairness tree.

    python -m edit_judge_bias.experiments.plot_results --metrics-dir results/v2/metrics
    python -m edit_judge_bias.experiments.plot_results \
        --metrics-dir results/v2_fairness_ds/metrics \
        --figures-dir results/v2_fairness_ds/figures

**The pilot prefix is a safety rule, not cosmetics.** The pilot archive under
``results/metrics`` is 1-5-scale data that includes ``mock-judge`` and the dead
``gpt-5.4-nano``; it used to render to the SAME basenames as the v2 grid
(``heatmap_mean_score_shift.png`` and friends). A ``\\includegraphics`` that resolved
to the wrong tree would have silently shipped mock-judge numbers in a paper about
frontier judges, with nothing on the page to give it away. Every figure written out of
a pilot-shaped tree now carries a ``pilot_`` basename, so the two trees can never
collide again.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.experiments.build_claim_tables import main_grid_rows
from edit_judge_bias.visualization import (
    paired_diff_sd,
    plot_bias_effect,
    plot_claim_a_dimension_forest,
    plot_claim_a_forest,
    plot_claim_b_forest,
    plot_ds_null_vs_control,
    plot_metric_heatmap,
    plot_mitigation_tradeoff,
    plot_position_joint,
    plot_quality_vs_floor,
    plot_retest,
    plot_rr_vs_cr,
)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def figure_prefix(metrics_dir: Path) -> str:
    """``"pilot_"`` for the 1-5 pilot archive, ``""`` for any v2-era tree.

    The pilot tree is ``<root>/results/metrics``; every later tree is one level deeper
    (``results/v2/metrics``, ``results/v2_fairness_ds/metrics``). So the parent
    directory's name is the discriminator, and it needs no config to stay right.
    """
    return "pilot_" if Path(metrics_dir).resolve().parent.name == "results" else ""


def make_figures(metrics_dir: Path, figures_dir: Path, *,
                 root: Path | None = None) -> List[Path]:
    metrics_dir, figures_dir = Path(metrics_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    pre = figure_prefix(metrics_dir)

    def out(name: str) -> Path:
        return figures_dir / f"{pre}{name}"

    scoring = _read_csv(metrics_dir / "scoring_shift.csv")
    if scoring:
        made.append(plot_metric_heatmap(
            scoring, "mean_shift", out("heatmap_mean_score_shift.png"),
            title="Mean score shift (biased - original)", center=0.0))
        made.append(plot_metric_heatmap(
            scoring, "sir", out("heatmap_score_inflation_rate.png"),
            title="Score inflation rate (P(biased > original))",
            cmap="Reds", center=None, fmt="{:.2f}"))
        made.append(plot_bias_effect(
            scoring, out("bias_effect_mean_shift.png"),
            value_key="mean_shift", title="Mean score shift by bias"))

    pairwise = _read_csv(metrics_dir / "pairwise_position.csv") or _read_csv(
        metrics_dir / "position.csv")
    if pairwise:
        made.append(plot_metric_heatmap(
            pairwise, "strict_flip_rate", out("heatmap_position_flip_rate.png"),
            title="Pairwise position strict flip rate", cmap="Reds", center=None))

    # ---- the frozen claim tables — present only once build_claim_tables has run ---
    # F2. claim_a.csv, never scoring_shift.csv: the latter has no CI, no BH q, no n and
    # no inside_placebo_bound, so it cannot separate an effect from the sham bound.
    claim_a = _read_csv(metrics_dir / "claim_a.csv")
    if claim_a:
        made.append(plot_claim_a_forest(claim_a, out("claim_a_forest.png")))

    # F13. The same cells split into the three dimensions the judge was asked for --
    # the only one of the figures that can separate "this cue degraded the picture"
    # from "this cue moved a score it cannot legitimately move". Drawn from
    # claim_a_by_dimension.csv, which is a decomposition of claim_a.csv on the same
    # complete-case cells, so the two figures can never disagree about a cell.
    claim_a_dims = _read_csv(metrics_dir / "claim_a_by_dimension.csv")
    if claim_a_dims:
        made.append(plot_claim_a_dimension_forest(
            claim_a_dims, out("claim_a_dimension_forest.png")))

    # F3. Grouped by cue. The judge-major `plot_agreement_forest` is deliberately NOT
    # rendered here: it is the same 40 estimates in an order that scatters the one
    # cue the claim is about, and two figures of the same numbers invite citing the
    # weaker one. It stays importable for ad-hoc use.
    # The main grid's 40 cells, as before 2026-09-15: the FILL rows merged into the same file
    # are measured against another reference, and one forest must not mix two references.
    claim_b = main_grid_rows(_read_csv(metrics_dir / "claim_b.csv"))
    if claim_b:
        made.append(plot_claim_b_forest(claim_b, out("claim_b_forest.png")))

    joint = _read_csv(metrics_dir / "position_joint.csv")
    if joint:
        made.append(plot_position_joint(joint, out("position_joint.png")))

    retest = _read_csv(metrics_dir / "retest.csv")
    if retest:
        made.append(plot_retest(retest, out("retest_noise_floor.png")))

    # F5. RR (slot swap) against CR (same question twice) — two different questions.
    position = _read_csv(metrics_dir / "position.csv")
    consistency = _read_csv(metrics_dir / "pairwise_consistency.csv")
    if position and consistency:
        made.append(plot_rr_vs_cr(position, consistency, out("position_rr_vs_cr.png")))

    # F7. What protocol-level mitigation buys and what it costs.
    swap = _read_csv(metrics_dir / "mitigation_swap_average.csv")
    if swap:
        made.append(plot_mitigation_tradeoff(
            swap, out("mitigation_swap_tradeoff.png")))

    # F8. Every cue against ITS OWN validator's false-flag floor.
    quality = _read_csv(metrics_dir / "quality_combined.csv")
    if quality and any(r.get("validator_model") for r in quality):
        made.append(plot_quality_vs_floor(quality, out("quality_vs_floor.png")))

    # F1. The D-S fairness arm — its own tree, its own tables.
    gaps = _read_csv(metrics_dir / "attribute_gaps.csv")
    deco = _read_csv(metrics_dir / "dose_control_decomposition.csv")
    if gaps:
        sd = {}
        base = root or Path(metrics_dir).resolve().parent.parent.parent
        try:
            sd = paired_diff_sd(
                base / "data" / "manifests" / "samples_fairness_ds_judge_v4.jsonl",
                Path(metrics_dir).parent / "raw_judgments",
            )
        except Exception as exc:  # pragma: no cover - the MDE ticks are optional
            print(f"note: MDE reference skipped ({exc})")
        made.append(plot_ds_null_vs_control(
            list(gaps) + list(deco), out("ds_null_vs_control.png"), paired_sd=sd))

    return made


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    ap = argparse.ArgumentParser(description="Render figures from metric CSVs.")
    ap.add_argument("--metrics-dir", type=Path, default=root / "results" / "v2" / "metrics")
    ap.add_argument("--figures-dir", type=Path, default=root / "results" / "v2" / "figures")
    args = ap.parse_args(argv)
    made = make_figures(args.metrics_dir, args.figures_dir, root=root)
    for p in made:
        print(f"wrote {p}")
    if not made:
        print("no metric CSVs found — run aggregate_results first")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
