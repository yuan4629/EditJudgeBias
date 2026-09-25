"""Guards for the WP-A1b three-axis dual-operator table.

The table exists because §5.4 ranks five judges on three layers and §7.2 prints three
between-layer correlations, and neither ranking was ever computed from a stated
operator. Four things can go wrong here and every one of them yields a CSV that looks
finished:

1. **the sign convention slips on one axis** — RR is "higher is better" while the
   per-item |change| is "higher is worse", so a single unflipped axis turns the
   inversion count into its complement;
2. **the placebo arm leaks into the absolute axis** — `sham` is the control the axis is
   read against, and counting it makes the control subject to the bound;
3. **ties get broken silently** — three judges sit at 9-of-12 significant cues, and
   alphabetical tie-breaking would manufacture two orderings that are not in the data;
4. **the protocol axis pretends to have a cell count** — it has one measurement per
   judge, so the second track substitutes a different statistic (CR), and that
   substitution has to stay visible in the row rather than be absorbed into a label.

A fifth since 2026-09-26: **the absolute axis falls back to a signed shift** — the
Method's invariance cell is `asc` per (cue, dimension) in `claim_a_by_dimension.csv`,
and both `mean_shift` columns and `claim_a.csv` sit right next to it, one wrong column
name away. The toy tree makes each of those readings order the judges the other way.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from edit_judge_bias.experiments.build_robustness_axes import (
    AXES,
    OPERATORS,
    TRACKS,
    axis_statistics,
    build_dual_operator_rows,
    build_operator_summary,
    count_inversions,
    rank_ascending,
)

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "results" / "v2" / "metrics"

on_disk = pytest.mark.skipif(
    not (METRICS / "claim_a.csv").exists(), reason="v2 claim tables not on disk"
)


# --------------------------------------------------------------------------- #
# a tiny synthetic metrics tree, so the extraction is pinned to cells not vibes #
# --------------------------------------------------------------------------- #
def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def toy_metrics(tmp_path: Path) -> Path:
    """Two judges, deliberately shaped so every guard below has something to catch.

    `alpha` is the more robust judge on the absolute axis but the less robust one on
    the protocol axis — the crossing §5.4 is about, in miniature. `sham` carries the
    biggest magnitude of all so any code path that forgets to drop it is caught.

    `beta`'s per-item changes are large but cancel in every cell (`mean_shift` 0.0),
    and `claim_a.csv`'s summed `mean_shift` makes `alpha` look worst: an absolute axis
    that read either signed shift would rank the two judges the other way round.
    """
    metrics = tmp_path / "metrics"
    _write(metrics / "claim_a.csv", [
        {"judge_model": "alpha", "bias_type": "sham", "mean_shift": "9.0",
         "significant_bh": "True"},
        {"judge_model": "alpha", "bias_type": "padding", "mean_shift": "-2.5",
         "significant_bh": "True"},
        {"judge_model": "alpha", "bias_type": "brightness", "mean_shift": "2.0",
         "significant_bh": "False"},
        {"judge_model": "beta", "bias_type": "sham", "mean_shift": "0.01",
         "significant_bh": ""},
        {"judge_model": "beta", "bias_type": "padding", "mean_shift": "-0.1",
         "significant_bh": "True"},
        {"judge_model": "beta", "bias_type": "brightness", "mean_shift": "0.05",
         "significant_bh": "True"},
    ])
    dim_rows = []
    for judge, cells in (
        # (cue, dimension, asc, mean_shift); sham asc 9.0 must never be counted
        ("alpha", [("sham", "instruction_adherence", "9.0", "9.0"),
                   ("sham", "detail_preservation", "9.0", "9.0"),
                   ("padding", "instruction_adherence", "0.5", "-0.5"),
                   ("padding", "detail_preservation", "0.3", "0.1"),
                   ("brightness", "instruction_adherence", "0.2", "0.2"),
                   ("brightness", "detail_preservation", "0.4", "-0.4")]),
        ("beta", [("sham", "instruction_adherence", "0.1", "0.0"),
                  ("sham", "detail_preservation", "0.1", "0.0"),
                  ("padding", "instruction_adherence", "2.0", "0.0"),
                  ("padding", "detail_preservation", "1.0", "0.0"),
                  ("brightness", "instruction_adherence", "1.5", "0.0"),
                  ("brightness", "detail_preservation", "1.1", "0.0")]),
    ):
        for cue, dim, asc, shift in cells:
            dim_rows.append({
                "judge_model": judge, "bias_type": cue, "dimension": dim,
                "mean_shift": shift, "asc": asc,
                "family": "control" if cue == "sham" else f"claim_A_{dim}",
            })
    _write(metrics / "claim_a_by_dimension.csv", dim_rows)
    _write(metrics / "claim_b.csv", [
        {"judge_model": "alpha", "bias_type": "padding", "spearman_delta": "-0.30",
         "rho_ci_excludes_zero": "True"},
        {"judge_model": "alpha", "bias_type": "brightness", "spearman_delta": "-0.01",
         "rho_ci_excludes_zero": "False"},
        {"judge_model": "beta", "bias_type": "padding", "spearman_delta": "-0.05",
         "rho_ci_excludes_zero": "True"},
        {"judge_model": "beta", "bias_type": "brightness", "spearman_delta": "-0.04",
         "rho_ci_excludes_zero": "True"},
    ])
    _write(metrics / "position.csv", [
        {"judge_model": "alpha", "n": "100", "rr": "0.60"},
        {"judge_model": "beta", "n": "100", "rr": "0.90"},
    ])
    _write(metrics / "pairwise_consistency.csv", [
        {"judge_model": "alpha", "n": "100", "cr": "0.95"},
        {"judge_model": "beta", "n": "100", "cr": "0.80"},
    ])
    return metrics


def test_absolute_axis_never_counts_the_placebo(toy_metrics):
    """`sham` is the control, not a cue: 9.0 must not become alpha's worst cell."""
    stats = axis_statistics(toy_metrics)
    assert stats[("worst_magnitude", "absolute")]["alpha"]["value"] == pytest.approx(0.5)
    assert stats[("worst_magnitude", "absolute")]["alpha"]["denominator"] == 4
    assert stats[("published_mean", "absolute")]["alpha"]["value"] == pytest.approx(0.35)
    assert stats[("published_mean", "absolute")]["alpha"]["denominator"] == 4
    # ... and the count operator must not count it either, even when it is stamped
    # significant (which the real table never does, but a future one might).
    assert stats[("significant_count", "absolute")]["alpha"]["value"] == 1.0


def test_absolute_axis_reads_the_per_item_change_not_a_signed_shift(toy_metrics):
    """beta's changes cancel inside every cell; the Method's invariance still sees them.

    Reading `mean_shift` from either table would call beta the more invariant judge —
    the operator this axis used until 2026-09-26, and the reason it was replaced.
    """
    stats = axis_statistics(toy_metrics)
    for track, beta_value in (("published_mean", 1.4), ("worst_magnitude", 2.0)):
        cells = stats[(track, "absolute")]
        assert cells["beta"]["value"] == pytest.approx(beta_value)
        assert cells["alpha"]["value"] < cells["beta"]["value"]
        assert cells["beta"]["unit"] == "rating points (1-10 scale)"
    for key in (("published_mean", "absolute"), ("worst_magnitude", "absolute")):
        assert "asc" in OPERATORS[key] and "dimension" in OPERATORS[key]


def test_protocol_axis_is_stored_as_a_failure_rate_not_a_success_rate(toy_metrics):
    """1 - RR and 1 - CR, so "bigger is worse" holds on all three axes."""
    stats = axis_statistics(toy_metrics)
    assert stats[("worst_magnitude", "protocol")]["alpha"]["value"] == pytest.approx(0.40)
    assert stats[("significant_count", "protocol")]["beta"]["value"] == pytest.approx(0.20)


def test_rank_one_is_the_most_robust_judge_on_every_axis(toy_metrics):
    rows = build_dual_operator_rows(toy_metrics)
    by = {(r["track"], r["axis"], r["judge_model"]): r for r in rows}
    # alpha deflates less in the worst cue -> rank 1 on the absolute axis;
    # alpha flips far more pairs -> rank 2 on the protocol axis. The lines cross.
    assert by[("worst_magnitude", "absolute", "alpha")]["rank"] == 1
    assert by[("worst_magnitude", "absolute", "beta")]["rank"] == 2
    assert by[("worst_magnitude", "protocol", "alpha")]["rank"] == 2
    assert by[("worst_magnitude", "protocol", "beta")]["rank"] == 1
    assert all(r["higher_is_worse"] for r in rows)


def test_every_row_carries_the_operator_that_produced_it(toy_metrics):
    rows = build_dual_operator_rows(toy_metrics)
    assert {(r["track"], r["axis"]) for r in rows} == set(OPERATORS)
    for r in rows:
        assert r["operator"] == OPERATORS[(r["track"], r["axis"])]
    # The one substitution in the design has to announce itself in the data.
    fallback = [r for r in rows
                if r["track"] == "significant_count" and r["axis"] == "protocol"]
    assert fallback and all("undefined" in r["note"] for r in fallback)
    assert all(r["note"] == "" for r in rows
               if not (r["track"] == "significant_count" and r["axis"] == "protocol"))


# --------------------------------------------------------------------------- #
# ranking and inversion arithmetic                                             #
# --------------------------------------------------------------------------- #
def test_ties_get_the_average_rank_and_are_not_inversions():
    """Three judges at the same count are one ordering, not three.

    This is not hypothetical: on the real table `gpt-5.5`, `kimi-k2.5` and
    `qwen3.5-plus` all sit at 9 significant cues out of 12.
    """
    ranks = rank_ascending({"a": 9.0, "b": 9.0, "c": 9.0, "d": 8.0, "e": 11.0})
    assert ranks["d"] == 1
    assert ranks["a"] == ranks["b"] == ranks["c"] == 3
    assert ranks["e"] == 5

    other = rank_ascending({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0})
    inv, tied, comp = count_inversions(ranks, other)
    assert comp == 10
    assert tied == 3           # the three pairs inside the tied block
    assert inv + tied <= comp  # a tied pair is never also an inversion


def test_missing_axis_values_drop_out_of_the_ranking_instead_of_sorting_as_zero():
    ranks = rank_ascending({"a": 0.5, "b": None, "c": 0.1})
    assert ranks["b"] is None
    assert ranks["c"] == 1 and ranks["a"] == 2
    inv, tied, comp = count_inversions(ranks, rank_ascending({"a": 1.0, "b": 2.0, "c": 3.0}))
    assert comp == 1  # only the a-c pair is comparable


def test_summary_reports_only_adjacent_pairs_as_adjacent(toy_metrics):
    summary = build_operator_summary(toy_metrics)
    pairs = {(r["track"], r["axis_pair"]): r for r in summary}
    assert pairs[("worst_magnitude", "absolute~ranking")]["adjacent"] is True
    assert pairs[("worst_magnitude", "ranking~protocol")]["adjacent"] is True
    assert pairs[("worst_magnitude", "absolute~protocol")]["adjacent"] is False
    pooled = pairs[("worst_magnitude", "(adjacent layers, pooled)")]
    adj = [r for r in summary
           if r["track"] == "worst_magnitude" and r["adjacent"]
           and not r["axis_pair"].startswith("(")]
    assert pooled["n_comparisons"] == sum(r["n_comparisons"] for r in adj)
    assert "independence expectation" in pooled["operator_second"]


# --------------------------------------------------------------------------- #
# the frozen tables                                                            #
# --------------------------------------------------------------------------- #
@on_disk
def test_frozen_tables_give_one_row_per_track_axis_judge():
    rows = build_dual_operator_rows(METRICS)
    judges = sorted({r["judge_model"] for r in rows})
    assert len(judges) == 5
    assert len(rows) == len(TRACKS) * len(AXES) * len(judges)
    for track in TRACKS:
        for axis in AXES:
            ranks = sorted(r["rank"] for r in rows
                           if r["track"] == track and r["axis"] == axis)
            assert None not in ranks, f"{track}/{axis} has an unranked judge"
            assert sum(ranks) == pytest.approx(15)  # 1+2+3+4+5, ties included


@on_disk
def test_published_track_reproduces_figure_5_rank_for_rank():
    """The published track must BE figure 5's ranking, not a lookalike.

    §7.2's three correlations and its inversion count are computed inside
    `plot_robustness_layers`. If this table's `published_mean` track ever drifts from
    that code, the paper would be quoting two different rankings under one name — which
    is the exact defect this whole module exists to expose, so it gets a test.
    """
    from edit_judge_bias.visualization.plot_robustness_layers import (
        absolute_score_robustness,
        position_robustness,
        rank_table,
        ranking_robustness,
    )
    from edit_judge_bias.experiments.build_claim_tables import main_grid_rows

    def read(name: str) -> list[dict]:
        with (METRICS / name).open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    figure = rank_table([
        absolute_score_robustness(read("claim_a_by_dimension.csv")),
        # claim_b.csv also holds the FILL v2 rows; figure 5 ranks the main grid's cells only.
        ranking_robustness(main_grid_rows(read("claim_b.csv"))),
        position_robustness(read("position.csv")),
    ])
    rows = {(r["axis"], r["judge_model"]): r
            for r in build_dual_operator_rows(METRICS) if r["track"] == "published_mean"}
    for judge, ranks in figure.items():
        for axis, rank in zip(AXES, ranks):
            assert rows[(axis, judge)]["rank"] == rank, f"{judge}/{axis}"


@on_disk
def test_published_track_pins_its_correlations_and_inversion_count():
    """+0.600 / -0.100 / -0.700 and 8 of 20 — each with its definition attached.

    Pinning them is what makes the other two tracks readable as a *sensitivity*:
    without an origin, a table of three tracks is three unrelated rankings. The
    +0.000 / -0.600 and 9 of 20 that §7.2 of the Chinese draft prints came from the
    |mean_shift| operator this axis used until 2026-09-26; ranking~protocol never
    involved the absolute axis and did not move.
    """
    summary = {r["axis_pair"]: r for r in build_operator_summary(METRICS)
               if r["track"] == "published_mean"}
    assert summary["absolute~ranking"]["spearman_rho"] == pytest.approx(0.600, abs=5e-4)
    assert summary["ranking~protocol"]["spearman_rho"] == pytest.approx(-0.100, abs=5e-4)
    assert summary["absolute~protocol"]["spearman_rho"] == pytest.approx(-0.700, abs=5e-4)
    pooled = summary["(adjacent layers, pooled)"]
    assert (pooled["n_inversions"], pooled["n_comparisons"]) == (8, 20)
    assert pooled["n_tied_pairs"] == 0


@on_disk
def test_absolute_axis_is_the_mean_and_max_of_the_36_asc_cells():
    """The Method's invariance, summarised two ways — and the frozen file agrees.

    The last loop catches the one failure no builder test can: code changed, tables
    not rebuilt through `scripts/06_build_tables.sh`.
    """
    with (METRICS / "claim_a_by_dimension.csv").open(encoding="utf-8", newline="") as fh:
        dim = [r for r in csv.DictReader(fh) if r["bias_type"] != "sham"]
    rows = build_dual_operator_rows(METRICS)
    by = {(r["track"], r["axis"], r["judge_model"]): r for r in rows}
    judges = sorted({r["judge_model"] for r in dim})
    assert len(judges) == 5
    for judge in judges:
        cells = [float(r["asc"]) for r in dim if r["judge_model"] == judge]
        assert len(cells) == 36  # 12 cues x 3 dimensions
        for track, value in (("published_mean", sum(cells) / len(cells)),
                             ("worst_magnitude", max(cells))):
            row = by[(track, "absolute", judge)]
            assert row["statistic"] == round(value, 4)
            assert row["denominator"] == 36

    with (METRICS / "three_axis_dual_operator.csv").open(encoding="utf-8", newline="") as fh:
        frozen = list(csv.DictReader(fh))
    assert len(frozen) == len(rows)
    for got, want in zip(frozen, rows):
        for key in ("track", "axis", "judge_model", "operator", "rank_trajectory"):
            assert got[key] == str(want[key]), (key, got, want)
        assert float(got["statistic"]) == want["statistic"]
        assert float(got["rank"]) == want["rank"]


@on_disk
def test_the_headline_trajectory_is_operator_sensitive_but_its_refutation_is_not():
    """The two highlighted judges' trajectories depend on the operator; the refutation does not.

    `gemini-3.5-flash` goes 5 -> 3 -> 1 (mean) and 5 -> 1 -> 1 (worst);
    `gpt-4o-viescore` goes 3 -> 5 -> 4 and 2 -> 5 -> 4 — it was 1 -> 5 -> 4 on both
    under the |mean_shift| operator used until 2026-09-26. And under all three tracks
    the scalar hypothesis (every rho = +1, zero inversions) fails. That asymmetry is
    the finding: the *refutation* is robust to the operator, the individual
    trajectories and correlations are not.
    """
    rows = {(r["track"], r["axis"], r["judge_model"]): r
            for r in build_dual_operator_rows(METRICS)}
    expected = {
        ("published_mean", "gemini-3.5-flash"): "5 -> 3 -> 1",
        ("worst_magnitude", "gemini-3.5-flash"): "5 -> 1 -> 1",
        ("published_mean", "gpt-4o-viescore"): "3 -> 5 -> 4",
        ("worst_magnitude", "gpt-4o-viescore"): "2 -> 5 -> 4",
    }
    for (track, judge), trajectory in expected.items():
        assert rows[(track, "absolute", judge)]["rank_trajectory"] == trajectory

    summary = build_operator_summary(METRICS)
    for track in TRACKS:
        rhos = [r["spearman_rho"] for r in summary
                if r["track"] == track and r["spearman_rho"] is not None]
        assert max(rhos) < 1.0, f"{track}: the scalar hypothesis would need rho = +1"
        pooled = next(r for r in summary
                      if r["track"] == track and r["axis_pair"].startswith("("))
        assert pooled["n_inversions"] > 0, f"{track}: the scalar hypothesis needs 0"


@on_disk
def test_the_published_correlations_swing_across_operators():
    """The same axis pair runs from -0.700 to +0.800 depending on the operator.

    Guards the sentence WP-B6 has to add: "not correlated" survives the choice of
    operator; the printed rho does not, and must never be quoted without its definition.
    """
    summary = build_operator_summary(METRICS)
    by_pair = {}
    for r in summary:
        if r["spearman_rho"] is not None:
            by_pair.setdefault(r["axis_pair"], []).append(r["spearman_rho"])
    spread = max(max(v) - min(v) for v in by_pair.values())
    assert spread > 0.5, "if the operator no longer matters, §7.2 can drop the caveat"


@on_disk
def test_the_published_reading_survives_in_the_magnitude_track():
    """`gemini-3.5-flash` last on the absolute axis and first on protocol under both
    magnitude operators; `gpt-4o-viescore` worst on the ranking axis.

    Those crossings are what §5.4 and figure 5 rest on, so they get a regression test
    at the level of the ranks rather than the prose.
    """
    rows = {(r["track"], r["axis"], r["judge_model"]): r
            for r in build_dual_operator_rows(METRICS)}
    for track in ("published_mean", "worst_magnitude"):
        assert rows[(track, "absolute", "gemini-3.5-flash")]["rank"] == 5
        assert rows[(track, "protocol", "gemini-3.5-flash")]["rank"] == 1
        assert rows[(track, "ranking", "gpt-4o-viescore")]["rank"] == 5
    assert rows[("worst_magnitude", "ranking", "gemini-3.5-flash")]["rank"] == 1
    # and the operator flip is real: on the count track the two swap ends.
    assert rows[("significant_count", "ranking", "gemini-3.5-flash")]["rank"] == 5
