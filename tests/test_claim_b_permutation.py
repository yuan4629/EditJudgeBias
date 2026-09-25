"""Guards for the WP-A1c cue-label permutation test.

The test asks whether "region_annotation wins 8 of 10 cells" is more than the best of
four cues looking best. A permutation test is only worth its null, and this null has
four properties that are easy to break and impossible to see in the output:

1. **one relabelling shared by all five judges.** Permuting each judge independently
   destroys the cross-judge coupling, so a coincidence that happens to hit every judge
   at once leaves the null — and the p-value collapses toward zero for the wrong
   reason. This is the single most consequential line in the module.
2. **each item keeps its own four scores.** If the shuffle moved scores between items,
   "some pictures degrade agreement more than others" would stop being an alternative
   explanation and start being evidence for the claim.
3. **an undefined correlation must not be counted as a win.** `argmin` over a row with
   a NaN silently elects a cue.
4. **the p-value must never come back exactly 0** from a finite number of draws.
"""

# ⚠️ The fixture calls below pass `roster=None` on purpose.  The default is
# `PUBLISHED_ROSTER` -- the five judges whose numbers are in the paper -- so that a
# forgotten argument silently OMITS a new judge (and says so) rather than silently
# ADMITTING one into a published table.  These tests use synthetic judge names, so they
# are about the mechanics and must opt out of family membership explicitly.


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_b_permutation import (
    CUES,
    _p_value,
    build_cells,
    build_permutation_rows,
    cell_deltas,
    permutation_null,
    spearman_columns,
    statistics_from_deltas,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "v2"

on_disk = pytest.mark.skipif(
    not (RESULTS / "metrics" / "claim_b.csv").exists(), reason="v2 claim tables not on disk"
)


# --------------------------------------------------------------------------- #
# a synthetic anchor arm                                                       #
# --------------------------------------------------------------------------- #
def _sample(sid: str, source: str, human: float, turn: str) -> SampleRecord:
    rec = SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add", content_category="object",
        original_image_path=f"{turn}.jpg", instruction=f"do {turn}", edit_model="m",
        edited_image_path=f"{sid}_e.jpg", human_score=human,
    )
    rec.metadata.anchor_source = source  # type: ignore[attr-defined]
    return rec


def _result(model: str, sid: str, fine: float, bias=None) -> JudgeResult:
    # `fine_score` is the sum of three integer dimensions (3-30), so the synthetic data
    # is quantised the same way the real grid is — which is also what puts ties in the
    # rank columns, the case `spearman_columns` has to get right.
    value = int(min(30, max(3, round(fine))))
    return JudgeResult(
        result_id=f"score::{model}::vanilla::{sid}::{bias}", judge_model=model,
        task_type="scoring", prompt_type="vanilla_scoring", raw_response_path="r.txt",
        parse_success=True, sample_id=sid, bias_type=bias,
        biased_id=f"{sid}__{bias}" if bias else None,
        overall_score=int(min(10, max(1, round(value / 3)))), fine_score=value,
        score_scale=10,
    )


def _build_tree(
    tmp_path: Path, *, damaged_cue: str | None, judges=("j1", "j2", "j3"),
    n_per_source: int = 32, seed: int = 7,
) -> tuple[Path, list[SampleRecord]]:
    """Three judges, two sources, four cues.

    Judges track the human ranking with independent noise. When `damaged_cue` is set,
    that cue's scores are replaced by an unrelated draw, so its rank correlation with
    the human column is destroyed while every other cue tracks the baseline.
    """
    rng = np.random.default_rng(seed)
    samples: list[SampleRecord] = []
    results = tmp_path / "v2"
    per_judge: dict[str, tuple[list, list]] = {m: ([], []) for m in judges}
    for source in ("srcA", "srcB"):
        for i in range(n_per_source):
            sid = f"{source}_{i:03d}"
            human = float(rng.uniform(0, 1))
            samples.append(_sample(sid, source, human, turn=f"{source}_t{i // 4}"))
            for model in judges:
                base = 15 + 12 * human + rng.normal(0, 1.2)
                per_judge[model][0].append(_result(model, sid, base))
                for cue in CUES:
                    if cue == damaged_cue:
                        value = 15 + rng.uniform(0, 12)
                    else:
                        value = base - 0.4 + rng.normal(0, 0.6)
                    per_judge[model][1].append(_result(model, sid, value, cue))
    for model, (originals, biased) in per_judge.items():
        io.write_jsonl(results / "raw_judgments" / f"scoring__{model}.jsonl", originals)
        io.write_jsonl(results / "biased_judgments" / f"scoring__{model}.jsonl", biased)
    return results, samples


# --------------------------------------------------------------------------- #
# the null's structural properties                                             #
# --------------------------------------------------------------------------- #
def test_one_relabelling_is_shared_by_every_judge(tmp_path: Path):
    """Two identical judges must stay identical under the permutation.

    If each judge were permuted independently their permuted cells would diverge, and
    the null would lose the cross-judge coincidences that are the main alternative
    explanation for 8-of-10.
    """
    results, samples = _build_tree(tmp_path, damaged_cue=None, judges=("j1",))
    # Clone j1 as j2 byte for byte, so any divergence downstream can only come from
    # the permutation being drawn twice.
    for kind in ("raw_judgments", "biased_judgments"):
        src = results / kind / "scoring__j1.jsonl"
        (results / kind / "scoring__j2.jsonl").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    cells, items = build_cells(samples, results, roster=None)
    rng = np.random.default_rng(0)
    permutation = rng.permuted(np.tile(np.arange(len(CUES)), (len(items), 1)), axis=1)
    by_judge: dict[str, list[np.ndarray]] = {}
    for cell in cells:
        by_judge.setdefault(cell.judge_model, []).append(cell_deltas(cell, permutation))
    for a, b in zip(by_judge["j1"], by_judge["j2"]):
        np.testing.assert_allclose(a, b)


def test_permutation_never_moves_a_score_between_items(tmp_path: Path):
    results, samples = _build_tree(tmp_path, damaged_cue=None, judges=("j1",))
    cells, items = build_cells(samples, results, roster=None)
    cell = cells[0]
    rng = np.random.default_rng(3)
    permutation = rng.permuted(np.tile(np.arange(len(CUES)), (len(items), 1)), axis=1)
    permuted = np.take_along_axis(cell.scores, permutation[cell.item_index], axis=1)
    np.testing.assert_allclose(np.sort(permuted, axis=1), np.sort(cell.scores, axis=1))


def test_items_are_the_complete_cases_across_every_judge(tmp_path: Path):
    """A cue missing for ONE judge removes the item from EVERY judge's cell."""
    results, samples = _build_tree(tmp_path, damaged_cue=None, judges=("j1", "j2"))
    path = results / "biased_judgments" / "scoring__j2.jsonl"
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if not ("srcA_000" in line and "padding" in line)]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    cells, items = build_cells(samples, results, roster=None)
    assert "srcA_000" not in items
    for cell in cells:
        assert cell.n_items == (31 if cell.anchor_source == "srcA" else 32)


# --------------------------------------------------------------------------- #
# the statistics                                                               #
# --------------------------------------------------------------------------- #
def test_an_undefined_cell_elects_no_cue():
    deltas = np.array([
        [-0.5, -0.1, -0.1, -0.1],      # cue 0 wins
        [np.nan, -0.9, -0.1, -0.1],    # undefined: must elect nobody, not cue 1
    ])
    stats = statistics_from_deltas(deltas)
    assert list(stats["worst_cue_cells"]) == [1.0, 0.0, 0.0, 0.0]
    assert stats["n_usable_cells"][0] == 1.0


def test_spearman_columns_matches_the_projects_own_implementation():
    from edit_judge_bias.metrics.agreement import spearman
    from scipy.stats import rankdata

    rng = np.random.default_rng(11)
    human = rng.integers(0, 20, size=40).astype(float)   # ties on purpose
    scores = rng.integers(3, 31, size=(40, 3)).astype(float)
    centred = rankdata(human) - rankdata(human).mean()
    unit = centred / np.linalg.norm(centred)
    got = spearman_columns(unit, scores)
    for c in range(3):
        assert got[c] == pytest.approx(spearman(list(human), list(scores[:, c])), abs=1e-9)


def test_p_value_is_the_add_one_estimator_and_never_returns_zero():
    null = np.arange(1000, dtype=float).reshape(-1, 1)
    assert _p_value(null, 1e9, "high") == pytest.approx(1 / 1001)
    assert _p_value(null, -1e9, "high") == pytest.approx(1.0)
    assert _p_value(null, 499.5, "high") == pytest.approx(501 / 1001, abs=1e-3)
    assert _p_value(null, -1e9, "low") == pytest.approx(1 / 1001)


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #
def test_a_planted_cue_is_detected_and_the_other_three_are_not(tmp_path: Path):
    results, samples = _build_tree(tmp_path, damaged_cue="region_annotation")
    rows = {(r["statistic"], r["bias_type"]): r
            for r in build_permutation_rows(samples, results, n_perm=200, roster=None)}
    for statistic in ("worst_cue_cells", "mean_delta_rho"):
        planted = rows[(statistic, "region_annotation")]
        assert planted["is_best_of_four"] is True
        assert planted["p_selected"] < 0.05
        for cue in CUES:
            if cue != "region_annotation":
                assert rows[(statistic, cue)]["p_selected"] > planted["p_selected"]


def test_with_no_planted_cue_nothing_comes_out_significant(tmp_path: Path):
    """Calibration. A permutation test that fires on exchangeable data is worthless.

    Seeds are fixed, so this is deterministic; it is a guard against a future change
    that quietly breaks exchangeability (permuting per judge, or permuting across
    items), each of which would make some cue look significant here.
    """
    results, samples = _build_tree(tmp_path, damaged_cue=None)
    rows = build_permutation_rows(samples, results, n_perm=200, roster=None)
    assert all(r["p_selected"] > 0.05 for r in rows)


def test_null_draws_are_centred_on_the_grand_mean(tmp_path: Path):
    """Under exchangeability every cue's null mean is the average of the four.

    A null centred anywhere else means the relabelling changed the data, not the label.
    """
    results, samples = _build_tree(tmp_path, damaged_cue="padding")
    cells, items = build_cells(samples, results, roster=None)
    observed = statistics_from_deltas(np.vstack([cell_deltas(c) for c in cells]))
    null = permutation_null(cells, len(items), n_perm=150, seed=1)
    grand = float(np.mean(observed["mean_delta_rho"]))
    for c in range(len(CUES)):
        assert null["mean_delta_rho"][:, c].mean() == pytest.approx(grand, abs=0.02)
    assert null["worst_cue_cells"].mean() == pytest.approx(len(cells) / len(CUES), abs=0.5)


# --------------------------------------------------------------------------- #
# the frozen tables                                                            #
# --------------------------------------------------------------------------- #
@on_disk
def test_frozen_grid_has_ten_cells_and_region_annotation_is_the_best_of_four():
    samples = io.read_jsonl(REPO / "data" / "manifests" / "samples_judge_v2.jsonl",
                            SampleRecord)
    rows = {(r["statistic"], r["bias_type"]): r
            for r in build_permutation_rows(samples, RESULTS, n_perm=100)}
    for statistic in ("worst_cue_cells", "mean_delta_rho"):
        row = rows[(statistic, "region_annotation")]
        assert row["n_cells"] == 10
        assert row["is_best_of_four"] is True
    counts = rows[("worst_cue_cells", "region_annotation")]
    # The three counts the module docstring reports, pinned so a future item-set change
    # cannot quietly move them: 8 under the published criterion, 8 under this
    # criterion on the published item sets, 7 on the complete cases.
    assert counts["published_n_ci_excludes_zero"] == 8
    assert counts["published_items_n_worst_of_four"] == 8
    assert counts["observed"] == 7.0
    assert counts["n_items_complete_case"] == 612
    assert counts["n_turns"] == 78
