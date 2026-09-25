"""The per-dimension invariance table read against the sham and retest noise floors."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult
from edit_judge_bias.experiments.build_noise_floor_table import (
    build_noise_floor_table,
    display,
    summarize,
)
from edit_judge_bias.metrics.scoring_metrics import retest_pairs
from edit_judge_bias.metrics.stats import bootstrap_ci, bootstrap_means

METRICS = Path(__file__).resolve().parents[1] / "results" / "v2" / "metrics"
JUDGE = "judge-x"
N = 20


def _score(sid, ia, eq, dp, bias=None, rep=1):
    return JudgeResult(
        result_id=f"{JUDGE}::{sid}::{bias}::{rep}", judge_model=JUDGE, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias,
        biased_id=f"{sid}__{bias}" if bias else None,
        bias_params={"repeat_index": rep} if rep > 1 else {},
        instruction_adherence=ia, editing_quality=eq, detail_preservation=dp,
        fine_score=ia + eq + dp, score_scale=10,
    )


def _write(tmp_path: Path, *, retest: bool = True) -> Path:
    """Baseline 5/5/5 everywhere. `sham` never moves (floor 0). The retest moves IA by 1
    on half the items (floor > 0 on IA only). `big` and `zoom_inset` lower every rating
    by 3; `small` lowers IA by 1 on a fifth of the items; `none` changes nothing."""
    results = tmp_path / "v2"
    raw = [_score(f"s{i}", 5, 5, 5) for i in range(N)]
    if retest:
        raw += [_score(f"s{i}", 6 if i % 2 else 5, 5, 5, rep=2) for i in range(N)]
    biased = []
    for i in range(N):
        sid = f"s{i}"
        biased += [
            _score(sid, 5, 5, 5, "sham"),
            _score(sid, 2, 2, 2, "big"),
            _score(sid, 2, 2, 2, "zoom_inset"),
            _score(sid, 4 if i % 5 == 0 else 5, 5, 5, "small"),
            _score(sid, 5, 5, 5, "none"),
        ]
    io.write_jsonl(results / "raw_judgments" / f"scoring__{JUDGE}.jsonl", raw)
    io.write_jsonl(results / "biased_judgments" / f"scoring__{JUDGE}.jsonl", biased)
    return results


def _cell(rows, name, dim):
    (row,) = [r for r in rows if r["row"] == name and r["dimension"] == dim]
    return row


def test_markers_count_the_floors_the_lower_bound_clears(tmp_path: Path):
    rows = build_noise_floor_table(_write(tmp_path), roster=None)
    ia = "instruction_adherence"
    assert _cell(rows, "sham", ia)["stat_x10"] == 0.0
    assert _cell(rows, "retest", ia)["stat_x10"] == 5.0
    assert _cell(rows, "big", ia)["marker"] == "bold"
    # clears the sham floor (0) but not the retest floor (~4)
    assert _cell(rows, "small", ia)["floors_cleared"] == 1
    assert _cell(rows, "small", ia)["marker"] == "plain"
    # strictly greater: a lower bound equal to a floor does not clear it
    assert _cell(rows, "none", ia)["marker"] == "gray"
    # on EQ the retest never moved, so both floors are 0 and `small` is gray there
    assert _cell(rows, "small", "editing_quality")["marker"] == "gray"


def test_floor_rows_carry_no_marker_and_every_cell_has_both_floors(tmp_path: Path):
    rows = build_noise_floor_table(_write(tmp_path), roster=None)
    for r in rows:
        if r["row_kind"] == "floor":
            assert r["marker"] is None and r["counted"] is None
        assert r["sham_floor"] is not None and r["retest_floor"] is not None


def test_borderline_cue_is_shown_but_not_counted(tmp_path: Path):
    rows = build_noise_floor_table(_write(tmp_path), roster=None)
    assert _cell(rows, "zoom_inset", "instruction_adherence")["marker"] == "bold"
    assert _cell(rows, "zoom_inset", "instruction_adherence")["counted"] is False
    s = summarize(rows)
    # `big` is bold on all three dimensions, `small` nowhere, `zoom_inset` not counted
    assert sum(s["bold_by_dimension"].values()) == 3
    assert s["bold_by_cue"]["zoom_inset"] == 3


def test_a_missing_floor_is_an_error_not_an_unmarked_table(tmp_path: Path):
    with pytest.raises(ValueError, match="retest"):
        build_noise_floor_table(_write(tmp_path, retest=False), roster=None)


def test_retest_pairs_reads_one_dimension():
    rows = [_score("s0", 5, 7, 5), _score("s0", 6, 7, 5, rep=2)]
    assert retest_pairs(rows, score_field="instruction_adherence") == {(JUDGE, "s0"): [5, 6]}
    assert retest_pairs(rows, score_field="editing_quality") == {(JUDGE, "s0"): [7, 7]}
    assert retest_pairs(rows) == {(JUDGE, "s0"): [17, 18]}   # fine_score by default


def test_bootstrap_means_is_the_distribution_bootstrap_ci_summarises():
    values = np.random.default_rng(0).integers(0, 4, size=200).astype(float)
    lo, hi = np.quantile(bootstrap_means(values), [0.025, 0.975])
    assert (float(lo), float(hi)) == bootstrap_ci(values)
    assert bootstrap_means([1.0]) is None


@pytest.mark.parametrize("value, shown", [
    (4.55, "4.6"), (10.25, "10.3"), (12.749590834697218, "12.7"), (0.0, "0.0"),
])
def test_display_rounds_half_up(value, shown):
    assert display(value) == shown


def test_published_counts():
    """Pinned on the built table; skipped when the judgments are not present."""
    path = METRICS / "invariance_noise_floor.csv"
    if not path.exists():
        pytest.skip("invariance_noise_floor.csv not built")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        r["counted"] = r["counted"] == "True"
    s = summarize(rows)
    assert s["bold_by_dimension"] == {"instruction_adherence": 27, "editing_quality": 34,
                                      "detail_preservation": 32}
    assert set(s["cells_by_dimension"].values()) == {55}
    assert s["bold_by_cue"]["saturation"] == 0
    cue_cells = [r for r in rows if r["row_kind"] == "cue"]
    assert len(cue_cells) == 180
