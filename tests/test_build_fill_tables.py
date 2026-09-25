"""Tests for the FILL v2 rows: every fill cell is measured against the fill's own sham, and
since 2026-09-15 the rows live inside the published claim tables.

Builder calls pass `roster=None` for the reason given at the top of
tests/test_build_claim_tables.py: the fixtures use made-up judge names, so these tests are
about mechanics and must opt out of family membership explicitly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import (
    BASELINE_LABEL,
    MAIN_COLLECTION,
    build_all,
    main_grid_rows,
)
from edit_judge_bias.experiments.build_fill_tables import (
    ANCHOR_CUES,
    COLLECTION,
    ENDORSED_SIDE_KEY,
    NOT_TABULATED,
    PAIRWISE_IMAGE_CUES,
    REFERENCE,
    REFERENCE_LABEL,
    build_collection_drift,
    build_fill_claim_b,
    build_fill_pairwise,
    with_endorsed_side,
)
from edit_judge_bias.metrics.pairwise_metrics import compute_one_sided_bias

REPO = Path(__file__).resolve().parents[1]


def _pw(model, pair_id, winner, bias=None, side=None, params=None):
    return JudgeResult(
        result_id=f"{model}::{pair_id}::{bias}::{side}", judge_model=model,
        task_type="pairwise", prompt_type="vanilla_pairwise", raw_response_path="r.txt",
        parse_success=True, pair_id=pair_id, winner=winner, bias_type=bias,
        biased_side=side, bias_params=params or {},
    )


def _score(model, sid, score, bias=None):
    return JudgeResult(
        result_id=f"{model}::{sid}::{bias}", judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sid, bias_type=bias, biased_id=f"{sid}__{bias}" if bias else None,
        overall_score=score, score_scale=10,
    )


def _sample(sid, *, block, source, human):
    rec = SampleRecord(
        sample_id=sid, source_dataset="src", edit_type="add", content_category="object",
        original_image_path=f"{sid}_o.jpg", instruction="do", edit_model="m",
        edited_image_path=f"{sid}_e.jpg", human_score=human,
    )
    rec.metadata.subset_block = block  # type: ignore[attr-defined]
    rec.metadata.anchor_source = source  # type: ignore[attr-defined]
    return rec


def _pairwise_fill(model, n, *, winner="B", side="b", target="A", overrides=None):
    """Every condition the pairwise fill asked, on pairs p0..p(n-1)."""
    overrides = overrides or {}
    rows = []
    for i in range(n):
        pid = f"p{i}"
        for bias in (REFERENCE,) + PAIRWISE_IMAGE_CUES:
            rows.append(_pw(model, pid, overrides.get(bias, winner), bias, side))
        rows.append(_pw(model, pid, overrides.get("bandwagon", winner), "bandwagon",
                        params={"bandwagon_target": target}))
        rows.append(_pw(model, pid, overrides.get("model_name", winner), "model_name",
                        params={"model_name_a": "x", "model_name_b": "y"}))
    return rows


def _write_pairwise(tmp_path, model, july_winner, fill_rows, n):
    v2, fill = tmp_path / "v2", tmp_path / "v2_fill"
    io.write_jsonl(v2 / "raw_judgments" / f"pairwise__{model}.jsonl",
                   [_pw(model, f"p{i}", july_winner) for i in range(n)])
    io.write_jsonl(fill / "biased_judgments" / f"pairwise__{model}.jsonl", fill_rows)
    return v2, fill


def _anchor_trees(tmp_path, *, n=8, july=None, sham=None, cue_scores=None):
    """n anchor-block samples plus one breadth-shared anchor sample (July rows only)."""
    july = july or (lambda i: n - i)          # July baseline: exactly reversed
    sham = sham or (lambda i: i + 1)          # fill sham: perfectly ranked
    cue_scores = cue_scores or {}
    samples = [_sample(f"s{i}", block="anchor", source="A", human=float(i)) for i in range(n)]
    samples.append(_sample("shared", block="breadth", source="A", human=3.5))
    v2, fill = tmp_path / "v2", tmp_path / "v2_fill"
    io.write_jsonl(v2 / "raw_judgments" / "scoring__m.jsonl",
                   [_score("m", f"s{i}", july(i)) for i in range(n)] + [_score("m", "shared", 5)])
    rows = []
    for i in range(n):
        rows.append(_score("m", f"s{i}", sham(i), REFERENCE))
        for cue in ANCHOR_CUES:
            rows.append(_score("m", f"s{i}", cue_scores.get(cue, sham)(i), cue))
    io.write_jsonl(fill / "biased_judgments" / "scoring__m.jsonl", rows)
    return samples, v2, fill


# --------------------------------------------------------------------------- #
# pairwise                                                                     #
# --------------------------------------------------------------------------- #
def test_a_fill_cue_is_read_against_the_fill_sham_not_the_july_baseline(tmp_path: Path):
    """The reason the module exists.  July: slot a wins every pair.  September: slot b wins
    every pair under EVERY condition, sham included -- pure between-collection drift and no
    cue effect at all.  Against July that reads as a perfect pull toward the dressed side;
    against the same run's sham it reads as nothing."""
    n = 8
    rows = _pairwise_fill("m", n, winner="B", side="b")
    v2, fill = _write_pairwise(tmp_path, "m", "A", rows, n)

    cells = {r["bias_type"]: r for r in build_fill_pairwise(v2, fill, roster=None)}
    assert cells["watermark"]["bias_advantage"] == 0.0
    assert cells["watermark"]["reference"] == REFERENCE_LABEL

    july = [_pw("m", f"p{i}", "A") for i in range(n)]
    [naive] = compute_one_sided_bias(july, [r for r in rows if r.bias_type == "watermark"])
    assert naive.bias_advantage == 1.0, "the drift the reference exists to cancel"


def test_the_reference_condition_is_never_tabulated_against_itself():
    rows = [_pw("m", "p0", "B", REFERENCE, "b"), _pw("m", "p0", "B", "padding", "b")]
    stats = compute_one_sided_bias(rows, rows, baseline_bias=REFERENCE)
    assert [st.bias_type for st in stats] == ["padding"]


def test_bandwagon_is_read_toward_the_slot_it_endorses(tmp_path: Path):
    n = 8
    rows = _pairwise_fill("m", n, winner="A", side="b", target="B", overrides={"bandwagon": "B"})
    v2, fill = _write_pairwise(tmp_path, "m", "A", rows, n)
    cells = {r["bias_type"]: r for r in build_fill_pairwise(v2, fill, roster=None)}
    assert cells["bandwagon"]["bias_advantage"] == 1.0
    assert cells["watermark"]["bias_advantage"] == 0.0


def test_model_name_has_no_one_sided_row_and_neither_does_the_reference(tmp_path: Path):
    n = 8
    v2, fill = _write_pairwise(tmp_path, "m", "A", _pairwise_fill("m", n), n)
    cues = {r["bias_type"] for r in build_fill_pairwise(v2, fill, roster=None)}
    assert cues == set(PAIRWISE_IMAGE_CUES) | set(ENDORSED_SIDE_KEY)
    assert "model_name" in NOT_TABULATED and REFERENCE not in cues


def test_fill_cells_carry_their_own_declared_families(tmp_path: Path):
    n = 8
    v2, fill = _write_pairwise(tmp_path, "m", "A", _pairwise_fill("m", n), n)
    [row] = [r for r in build_fill_pairwise(v2, fill, roster=None) if r["bias_type"] == "zoom_inset"]
    assert row["family"] == "pairwise_one_sided_advantage + _accuracy [fill]"
    assert "q_value_advantage" in row and "q_value_accuracy" in row
    assert "q_value" not in row, "an unqualified q_value would not say which family"


def test_a_half_collected_condition_refuses_to_build(tmp_path: Path):
    n = 8
    rows = [r for r in _pairwise_fill("m", n)
            if not (r.bias_type == "distraction" and r.pair_id in {"p0", "p1", "p2", "p3"})]
    v2, fill = _write_pairwise(tmp_path, "m", "A", rows, n)
    with pytest.raises(ValueError, match="incomplete"):
        build_fill_pairwise(v2, fill, roster=None)


def test_a_missing_fill_tree_is_an_error_not_an_empty_table(tmp_path: Path):
    v2 = tmp_path / "v2"
    io.write_jsonl(v2 / "raw_judgments" / "pairwise__m.jsonl", [_pw("m", "p0", "A")])
    with pytest.raises(FileNotFoundError, match="v2_fill"):
        build_fill_pairwise(v2, tmp_path / "v2_fill", roster=None)


def test_an_endorsed_slot_that_is_missing_or_contradicted_raises():
    with pytest.raises(ValueError, match="bandwagon_target"):
        with_endorsed_side([_pw("m", "p0", "A", "bandwagon", params={})])
    with pytest.raises(ValueError, match="contradicts"):
        with_endorsed_side([_pw("m", "p0", "A", "bandwagon", "a", {"bandwagon_target": "B"})])


# --------------------------------------------------------------------------- #
# claim B                                                                     #
# --------------------------------------------------------------------------- #
def test_claim_b_fill_takes_its_before_column_from_the_fill_sham(tmp_path: Path):
    samples, v2, fill = _anchor_trees(tmp_path, cue_scores={"watermark": lambda i: 8 - i})
    rows = build_fill_claim_b(samples, v2, fill, n_boot=50, roster=None)
    [wm] = [r for r in rows if r["bias_type"] == "watermark"]
    assert wm["spearman_original"] == 1.0, "before = the fill sham, not the reversed July baseline"
    assert wm["spearman_biased"] == -1.0
    assert wm["n_paired"] == 8, "the breadth-shared sample has no fill sham and stays out"
    assert wm["family"] == "claim_B_acc [fill]"
    assert wm["reference"] == REFERENCE_LABEL


def test_a_fill_row_outside_the_anchor_only_design_is_refused(tmp_path: Path):
    samples, v2, fill = _anchor_trees(tmp_path)
    path = fill / "biased_judgments" / "scoring__m.jsonl"
    io.write_jsonl(path, io.read_jsonl(path, JudgeResult) + [_score("m", "shared", 5, REFERENCE)])
    with pytest.raises(ValueError, match="outside the anchor-only design"):
        build_fill_claim_b(samples, v2, fill, n_boot=20, roster=None)


# --------------------------------------------------------------------------- #
# drift + driver                                                              #
# --------------------------------------------------------------------------- #
def test_collection_drift_compares_the_fill_sham_with_the_july_baseline(tmp_path: Path):
    n = 8
    samples, v2, fill = _anchor_trees(tmp_path, n=n, july=lambda i: i + 1, sham=lambda i: i + 3)
    io.write_jsonl(v2 / "raw_judgments" / "pairwise__m.jsonl",
                   [_pw("m", f"p{i}", "A") for i in range(n)])
    io.write_jsonl(fill / "biased_judgments" / "pairwise__m.jsonl",
                   _pairwise_fill("m", n, winner="B", side="b"))
    [row] = build_collection_drift(samples, v2, fill, roster=None)
    assert row["scoring_shift"] == 2.0 and row["scoring_n"] == n
    assert row["pairwise_shift"] == 1.0 and row["pairwise_n"] == n


def _merged_trees(tmp_path: Path, n: int = 8):
    """A main grid (one anchor cue, one one-sided cue) beside a complete fill, for `build_all`."""
    samples, v2, fill = _anchor_trees(tmp_path, n=n, cue_scores={"zoom_inset": lambda i: n - i})
    io.write_jsonl(v2 / "biased_judgments" / "scoring__m.jsonl",
                   [_score("m", s.sample_id, (k % 3) + 1, "padding") for k, s in enumerate(samples)])
    io.write_jsonl(v2 / "raw_judgments" / "pairwise__m.jsonl",
                   [_pw("m", f"p{i}", "A") for i in range(n)])
    io.write_jsonl(v2 / "biased_judgments" / "pairwise__m.jsonl",
                   [_pw("m", f"p{i}", "B" if i % 2 else "A", "padding", "b") for i in range(n)])
    io.write_jsonl(fill / "biased_judgments" / "pairwise__m.jsonl", _pairwise_fill("m", n))
    samples_path = tmp_path / "samples.jsonl"
    io.write_jsonl(samples_path, samples)
    return samples_path, v2, fill


def _csv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_fill_rows_live_in_the_published_files_and_say_where_they_came_from(tmp_path: Path):
    """User decision, 2026-09-15: no side files.  Each row names its collection and its
    reference, the fill keeps its own families, and the two columns sit before `family`."""
    samples_path, v2, fill = _merged_trees(tmp_path)
    build_all(v2, samples_path=samples_path, roster=None, n_boot=20, fill_dir=fill)
    metrics = v2 / "metrics"
    assert not list(metrics.glob("*_fill.csv")), "the side files are retired"
    assert (metrics / "fill_collection_drift.csv").exists()
    families = {
        "claim_b.csv": ("claim_B_acc", "claim_B_acc [fill]"),
        "pairwise_one_sided.csv": ("pairwise_one_sided_advantage + _accuracy",
                                   "pairwise_one_sided_advantage + _accuracy [fill]"),
    }
    for name, (main_family, fill_family) in families.items():
        rows = _csv(metrics / name)
        got = {(r["collection"], r["reference"], r["family"]) for r in rows}
        assert got == {(MAIN_COLLECTION, BASELINE_LABEL, main_family),
                       (COLLECTION, REFERENCE_LABEL, fill_family)}, name
        header = list(rows[0])
        assert header.index("reference") + 1 == header.index("collection") == header.index("family") - 1


def test_merging_the_fill_moves_no_main_grid_value(tmp_path: Path):
    """The merge only adds rows and two provenance columns: every main-grid cell, q-values
    included, is exactly what the same build writes without the fill."""
    samples_path, v2, fill = _merged_trees(tmp_path)
    alone = build_all(v2, samples_path=samples_path, roster=None, n_boot=20,
                      out_dir=tmp_path / "alone")
    merged = build_all(v2, samples_path=samples_path, roster=None, n_boot=20, fill_dir=fill,
                       out_dir=tmp_path / "merged")

    def canon(rows):
        return [json.dumps({k: v for k, v in r.items() if k not in ("reference", "collection")},
                           sort_keys=True, default=str) for r in rows]

    for key in ("claim_b", "one_sided"):
        assert alone[key], key
        assert canon(main_grid_rows(merged[key])) == canon(alone[key]), key
        assert len(merged[key]) > len(alone[key]), key


def test_a_second_roster_tags_its_merged_fill_rows_too(tmp_path: Path):
    samples_path, v2, fill = _merged_trees(tmp_path)
    tables = build_all(v2, samples_path=samples_path, roster=None, n_boot=20, fill_dir=fill,
                       prefix="a5_")
    metrics = v2 / "metrics"
    for name in ("claim_b", "pairwise_one_sided", "fill_collection_drift"):
        assert (metrics / f"a5_{name}.csv").exists(), name
        assert not (metrics / f"{name}.csv").exists(), "a second roster never takes the published name"
    fill_families = {r["family"] for key in ("claim_b", "one_sided")
                     for r in tables[key] if r["collection"] == COLLECTION}
    assert fill_families == {"claim_B_acc [fill] [a5 roster]",
                             "pairwise_one_sided_advantage + _accuracy [fill] [a5 roster]"}


def test_main_grid_rows_keeps_rows_without_a_collection_and_drops_the_fill():
    rows = [{"bias_type": "padding"}, {"bias_type": "text_overlay", "collection": MAIN_COLLECTION},
            {"bias_type": "zoom_inset", "collection": COLLECTION}]
    assert [r["bias_type"] for r in main_grid_rows(rows)] == ["padding", "text_overlay"]


def test_a_named_fill_dir_that_does_not_exist_is_an_error(tmp_path: Path):
    samples_path, v2, _ = _merged_trees(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_all(v2, samples_path=samples_path, roster=None, n_boot=20,
                  fill_dir=tmp_path / "nowhere")


def test_the_cli_refuses_to_rebuild_results_v2_without_its_fill(tmp_path: Path, capsys):
    """The failure it prevents is silent: a hand rebuild without `--fill-dir` would write
    claim tables that no longer hold the cells main_table.csv counts."""
    from edit_judge_bias.experiments import build_claim_tables as bct

    with pytest.raises(SystemExit):
        bct.main(["--results-dir", str(REPO / "results" / "v2"), "--out-dir", str(tmp_path)])
    assert "--fill-dir" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(not (REPO / "configs" / "experiment" / "pairwise_fill_v2.yaml").exists(),
                    reason="fill configs not present")
def test_the_declared_cues_are_exactly_what_the_fill_arms_asked():
    """The builder's cue lists are constants; the arms' configs are what was paid for.  If
    the two ever disagree, a condition is being silently dropped or invented."""
    pw = yaml.safe_load((REPO / "configs/experiment/pairwise_fill_v2.yaml").read_text(encoding="utf-8"))
    asked = set(pw["one_sided_biases"]["bias_types"]) | {s["bias_type"] for s in pw["prompt_biases"]}
    assert asked == {REFERENCE} | set(PAIRWISE_IMAGE_CUES) | set(ENDORSED_SIDE_KEY) | set(NOT_TABULATED)
    an = yaml.safe_load((REPO / "configs/experiment/scoring_anchor_fill_v2.yaml").read_text(encoding="utf-8"))
    asked = set(an["bias_types"]) | {s["bias_type"] for s in an["prompt_biases"]}
    assert asked == {REFERENCE} | set(ANCHOR_CUES)


FROZEN = REPO / "results" / "v2" / "metrics"


@pytest.mark.skipif(not (FROZEN / "claim_b.csv").exists(), reason="v2 claim tables not on disk")
def test_the_frozen_claim_tables_hold_both_collections_each_in_its_own_family():
    """What the merge produced on the real tree: row counts per collection, one family and one
    reference per collection, and no side file left behind."""
    pw = "pairwise_one_sided_advantage + _accuracy"
    expect = {
        "claim_b.csv": {MAIN_COLLECTION: (40, "claim_B_acc"), COLLECTION: (80, "claim_B_acc [fill]")},
        "pairwise_one_sided.csv": {MAIN_COLLECTION: (15, pw), COLLECTION: (40, pw + " [fill]")},
        "a5_claim_b.csv": {MAIN_COLLECTION: (8, "claim_B_acc [a5 roster]"),
                           COLLECTION: (16, "claim_B_acc [fill] [a5 roster]")},
        "a5_pairwise_one_sided.csv": {MAIN_COLLECTION: (3, pw + " [a5 roster]"),
                                      COLLECTION: (8, pw + " [fill] [a5 roster]")},
    }
    reference = {MAIN_COLLECTION: BASELINE_LABEL, COLLECTION: REFERENCE_LABEL}
    for name, want in expect.items():
        got = {}
        for r in _csv(FROZEN / name):
            got.setdefault(r["collection"], []).append(r)
        assert {c: (len(v), {x["family"] for x in v}, {x["reference"] for x in v})
                for c, v in got.items()} == {
            c: (n, {fam}, {reference[c]}) for c, (n, fam) in want.items()}, name
        assert not (FROZEN / name.replace(".csv", "_fill.csv")).exists(), name


P4_TABLE = REPO / "results" / "v2_fill" / "metrics" / "quality_combined.csv"


@pytest.mark.skipif(not P4_TABLE.exists(), reason="FILL v2 P4 table not built")
def test_the_p4_validator_table_covers_its_design_and_stays_out_of_the_published_one():
    """P4 = the two gating validators x (7 one-sided image cues + sham) on 110 fill bases.
    A partial collection must not pass for a finished one, and none of its rows may land in
    the published quality table, whose per-(validator, cue) rows other code indexes."""
    rows = _csv(P4_TABLE)
    assert {r["validator_model"] for r in rows} == {"gemini-3.5-flash", "gpt-4o-mini"}
    assert {r["bias_type"] for r in rows} == set(PAIRWISE_IMAGE_CUES) | {REFERENCE}
    for r in rows:
        assert int(r["n_mllm"]) >= 105, (r["validator_model"], r["bias_type"], r["n_mllm"])
        assert r["n_human"] == "0", "no human leg was drawn from these pictures"
    published = _csv(REPO / "results" / "v2" / "metrics" / "quality_combined.csv")
    assert len({(r["validator_model"], r["bias_type"]) for r in published}) == len(published)
    assert len(published) == 33, "3 validators x 11 cues; a P4 row here would add one"
