"""Tests for the per-dimension agreement table (the main table's Delta-rho markers).

Synthetic fixtures only; builder calls pass `roster=None` because the judge name is made
up (see tests/test_build_claim_tables.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_agreement_by_dimension import (
    FAMILY,
    HUMAN_LABEL,
    build_agreement_by_dimension,
    delta_rho_cell,
    labelled_sources,
    main,
    summarize,
)
from edit_judge_bias.experiments.build_claim_tables import ANCHOR_BIASES, BASELINE_LABEL
from edit_judge_bias.experiments.build_fill_tables import (
    ANCHOR_CUES as FILL_CUES,
    REFERENCE,
    REFERENCE_LABEL,
)
from edit_judge_bias.metrics.agreement import (
    bootstrap_p_two_sided,
    cluster_bootstrap_ci,
    cluster_bootstrap_values,
)

N_TURNS, EDITORS = 6, 3
DIMS = ("instruction_adherence", "editing_quality", "detail_preservation")


def _sample(i: int, source: str, *, labelled: bool = True) -> SampleRecord:
    turn = i // EDITORS
    rec = SampleRecord(
        sample_id=f"{source}{i}", source_dataset=source, edit_type="add",
        content_category="object", original_image_path=f"{source}_turn{turn}.jpg",
        instruction=f"do {turn}", edit_model=f"e{i % EDITORS}",
        edited_image_path=f"{source}{i}_e.jpg", human_score=float(i),
    )
    rec.metadata.anchor_source = source  # type: ignore[attr-defined]
    rec.metadata.subset_block = "anchor"  # type: ignore[attr-defined]
    if labelled:
        for label in HUMAN_LABEL.values():
            setattr(rec.metadata, label, float(i))
    return rec


def _score(sid: str, dims: dict, bias=None, model="m") -> JudgeResult:
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias, biased_id=f"{sid}__{bias}" if bias else None,
        score_scale=10, fine_score=sum(dims.values()), **dims,
    )


def _dims(i: int, *, reverse=()) -> dict:
    n = N_TURNS * EDITORS
    return {d: (1 + (n - 1 - i) * 9 // (n - 1)) if d in reverse else (1 + i * 9 // (n - 1))
            for d in DIMS}


def _trees(tmp_path: Path, *, july_reverse=(), main_cue=None, fill_cue=None):
    """One labelled source ("A") and one without per-dimension labels ("B")."""
    n = N_TURNS * EDITORS
    samples = [_sample(i, "A") for i in range(n)] + [_sample(i, "B", labelled=False)
                                                     for i in range(n)]
    v2, fill = tmp_path / "v2", tmp_path / "v2_fill"
    raw, biased, filled = [], [], []
    for s in samples:
        i = int(s.sample_id[1:])
        raw.append(_score(s.sample_id, _dims(i, reverse=july_reverse)))
        for cue in ANCHOR_BIASES:
            rev = main_cue[1] if main_cue and main_cue[0] == cue else july_reverse
            biased.append(_score(s.sample_id, _dims(i, reverse=rev), cue))
        filled.append(_score(s.sample_id, _dims(i), REFERENCE))
        for cue in FILL_CUES:
            rev = fill_cue[1] if fill_cue and fill_cue[0] == cue else ()
            filled.append(_score(s.sample_id, _dims(i, reverse=rev), cue))
    io.write_jsonl(v2 / "raw_judgments" / "scoring__m.jsonl", raw)
    io.write_jsonl(v2 / "biased_judgments" / "scoring__m.jsonl", biased)
    io.write_jsonl(fill / "biased_judgments" / "scoring__m.jsonl", filled)
    return samples, v2, fill


def _cells(rows):
    return {(r["bias_type"], r["dimension"]): r for r in rows}


# --------------------------------------------------------------------------- #
# the p-value and the interval share their resamples                         #
# --------------------------------------------------------------------------- #
def test_bootstrap_p_counts_the_replicates_that_reach_the_null():
    assert bootstrap_p_two_sided([0.1] * 9) == pytest.approx(2 * 1 / 10)
    assert bootstrap_p_two_sided([-1.0, 1.0] * 50) == 1.0
    assert bootstrap_p_two_sided([0.5]) is None
    # 3 of 99 replicates at or below zero: 2 * (3 + 1) / 100
    assert bootstrap_p_two_sided([-0.1] * 3 + [0.2] * 96) == pytest.approx(0.08)


def test_the_interval_is_the_percentiles_of_the_exposed_replicates():
    rng = np.random.default_rng(0)
    clusters = [list(rng.normal(size=4)) for _ in range(12)]
    stat = lambda xs: float(np.mean(xs))  # noqa: E731
    values = cluster_bootstrap_values(clusters, stat, n_boot=300, seed=7)
    lo, hi = cluster_bootstrap_ci(clusters, stat, n_boot=300, seed=7)
    assert (lo, hi) == tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))
    assert cluster_bootstrap_values(clusters[:1], stat) == []


# --------------------------------------------------------------------------- #
# one cell                                                                    #
# --------------------------------------------------------------------------- #
def test_each_dimension_is_read_against_its_own_label_not_the_summed_score(tmp_path):
    """Instruction adherence reverses, detail preservation does not move: the two cells of
    the same (judge, cue) must disagree. A summed score would give them one value."""
    samples, v2, fill = _trees(tmp_path, fill_cue=("watermark", ("instruction_adherence",)))
    fill_rows = io.read_jsonl(fill / "biased_judgments" / "scoring__m.jsonl", JudgeResult)
    subset = [s for s in samples if s.metadata.anchor_source == "A"]
    ref = [r for r in fill_rows if r.bias_type == REFERENCE]
    cued = [r for r in fill_rows if r.bias_type == "watermark"]
    ia = delta_rho_cell(subset, ref, cued, judge_model="m", bias_type="watermark",
                        dimension="instruction_adherence", n_boot=200)
    dp = delta_rho_cell(subset, ref, cued, judge_model="m", bias_type="watermark",
                        dimension="detail_preservation", n_boot=200)
    assert ia["human_label"] == "mos_alignment" and dp["human_label"] == "mos_preservation"
    assert ia["spearman_delta"] < -1.9          # the rating reverses; ties keep it off -2
    assert ia["marker"] == "down" and ia["p_boot"] == pytest.approx(2 / 201)
    assert dp["spearman_delta"] == pytest.approx(0.0) and dp["marker"] == ""
    assert ia["n_paired"] == dp["n_paired"] == N_TURNS * EDITORS
    assert ia["n_clusters"] == N_TURNS


def test_only_sources_with_all_three_labels_get_rows(tmp_path):
    samples, _, _ = _trees(tmp_path)
    assert list(labelled_sources(samples)) == ["A"]


# --------------------------------------------------------------------------- #
# the table                                                                   #
# --------------------------------------------------------------------------- #
def test_a_fill_cue_is_read_against_the_fill_sham_and_a_main_cue_against_july(tmp_path):
    """July's baseline ranks every dimension backwards (drift between collections); the
    fill's sham and cues rank it correctly. The fill cue must read as no change (against
    its own sham), and a main-grid cue identical to July must read as no change too."""
    samples, v2, fill = _trees(tmp_path, july_reverse=DIMS)
    rows = build_agreement_by_dimension(samples, v2, fill, roster=None, n_boot=200)
    cells = _cells(rows)
    assert len(rows) == (len(ANCHOR_BIASES) + len(FILL_CUES)) * len(DIMS)
    for d in DIMS:
        assert cells[("watermark", d)]["spearman_delta"] == pytest.approx(0.0)
        assert cells[("watermark", d)]["reference"] == REFERENCE_LABEL
        assert cells[("padding", d)]["spearman_delta"] == pytest.approx(0.0)
        assert cells[("padding", d)]["reference"] == BASELINE_LABEL
    assert {r["anchor_source"] for r in rows} == {"A"}


def test_bh_is_added_beside_the_published_marker_and_never_replaces_it(tmp_path):
    samples, v2, fill = _trees(tmp_path, main_cue=("padding", ("editing_quality",)),
                               fill_cue=("zoom_inset", ("detail_preservation",)))
    rows = build_agreement_by_dimension(samples, v2, fill, roster=None, n_boot=200)
    cells = _cells(rows)
    assert cells[("padding", "editing_quality")]["marker"] == "down"
    assert cells[("zoom_inset", "detail_preservation")]["marker"] == "down"
    assert {r["family"] for r in rows} == {FAMILY}
    assert {r["family_by_collection"] for r in rows} == {FAMILY + " [main grid]",
                                                        FAMILY + " [fill]"}
    for r in rows:
        assert r["q_value"] >= r["p_boot"] - 1e-9
        assert r["marker_bh"] in ("", r["marker"])
        assert r["marker_bh_by_collection"] in ("", r["marker"])
        assert r["counted"] == (r["bias_type"] != "zoom_inset")
    s = summarize(rows)
    assert s["marker"]["all"] == {"cells": 33, "down": 1, "up": 0}   # zoom_inset not counted
    assert s["marker"]["site=pixel"]["down"] == 1


def test_a_missing_fill_tree_is_an_error_not_a_quiet_skip(tmp_path):
    samples, v2, _ = _trees(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_agreement_by_dimension(samples, v2, tmp_path / "absent", roster=None, n_boot=50)


def test_the_command_line_writes_one_row_per_cell(tmp_path, capsys):
    samples, v2, fill = _trees(tmp_path, fill_cue=("bandwagon", ("editing_quality",)))
    manifest = tmp_path / "samples.jsonl"
    io.write_jsonl(manifest, samples)
    rc = main(["--results-dir", str(v2), "--fill-dir", str(fill), "--samples", str(manifest),
               "--out-dir", str(tmp_path / "out"), "--n-boot", "100", "--roster", "all"])
    assert rc == 0
    text = (tmp_path / "out" / "all_agreement_by_dimension.csv").read_text(encoding="utf-8")
    assert len(text.strip().splitlines()) == 1 + (len(ANCHOR_BIASES) + len(FILL_CUES)) * 3
    assert "UNCORRECTED" in capsys.readouterr().out
