"""Tests for the full-version manifest combiner.

The two invariants worth guarding: the seeded draw must be a *prefix* (so raising
the subset size never invalidates already-injected images or already-paid judge
calls), and the per-builder edit_type trust flags must normalise to one key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from edit_judge_bias.data.build_full_manifest import (
    combine,
    decisive_tier,
    edit_type_is_trusted,
    load_config,
)
from edit_judge_bias.data.io import read_jsonl, write_jsonl
from edit_judge_bias.data.schema import PairRecord, SampleRecord


def _img(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "img" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def _sample(tmp_path: Path, sid: str, *, source: str, edit_type: str, extra=None) -> SampleRecord:
    orig = _img(tmp_path, f"{sid}_o.png")
    edit = _img(tmp_path, f"{sid}_e.png")
    return SampleRecord(
        sample_id=sid,
        source_dataset=source,
        edit_type=edit_type,
        content_category="object",
        original_image_path=orig.relative_to(tmp_path),
        edited_image_path=edit.relative_to(tmp_path),
        instruction="add a cup",
        edit_model=f"m_{sid[-1]}",
        metadata=(extra or {}),
    )


def _pair(tmp_path: Path, pid: str, a: str, b: str, *, source: str, edit_type: str, gt=None):
    return PairRecord(
        pair_id=pid,
        sample_id_a=a,
        sample_id_b=b,
        original_image_path=_img(tmp_path, f"{pid}_o.png").relative_to(tmp_path),
        edited_image_a_path=_img(tmp_path, f"{pid}_a.png").relative_to(tmp_path),
        edited_image_b_path=_img(tmp_path, f"{pid}_b.png").relative_to(tmp_path),
        instruction="add a cup",
        edit_model_a="m_a",
        edit_model_b="m_b",
        ground_truth_preference=gt,
        source_dataset=source,
        edit_type=edit_type,
    )


def _fixture(tmp_path: Path, *, k=2, pair_k=2, require_trusted=True, pairs_gt_only=False):
    """Two sources x two edit_types x 4 samples, plus pairs (half with human GT)."""
    samples, pairs = [], []
    for source in ("SrcA", "SrcB"):
        for et in ("add", "remove"):
            ids = [f"{source[-1]}{et[0]}{i}" for i in range(4)]
            for i, sid in enumerate(ids):
                # One row per stratum is default-labelled, i.e. untrusted.
                extra = {"edit_type_source": "default"} if i == 3 else {"edit_type_source": "rules"}
                samples.append(_sample(tmp_path, sid, source=source, edit_type=et, extra=extra))
            for i in range(2):
                pairs.append(
                    _pair(
                        tmp_path,
                        f"p_{source}_{et}_{i}",
                        ids[i],
                        ids[i + 1],
                        source=source,
                        edit_type=et,
                        gt="a" if i == 0 else None,
                    )
                )
    write_jsonl(tmp_path / "s1.jsonl", [s for s in samples if s.source_dataset == "SrcA"])
    write_jsonl(tmp_path / "s2.jsonl", [s for s in samples if s.source_dataset == "SrcB"])
    write_jsonl(tmp_path / "p1.jsonl", [p for p in pairs if p.source_dataset == "SrcA"])
    write_jsonl(tmp_path / "p2.jsonl", [p for p in pairs if p.source_dataset == "SrcB"])

    cfg = {
        "sources": [
            {"name": "SrcA", "samples": "s1.jsonl", "pairs": "p1.jsonl"},
            {"name": "SrcB", "samples": "s2.jsonl", "pairs": "p2.jsonl"},
        ],
        "output": {"samples": "out_s.jsonl", "pairs": "out_p.jsonl"},
        "judge_subset": {
            "seed": 42,
            "per_source_per_edit_type": k,
            "pairs_per_source_per_edit_type": pair_k,
            "require_edit_type_trusted": require_trusted,
            "pairs_require_ground_truth": pairs_gt_only,
            "samples": "sub_s.jsonl",
            "pairs": "sub_p.jsonl",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


# --------------------------------------------------------------------------- #
# edit_type trust normalisation                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, True),  # I2EBench ships an authoritative label and no flag
        ({"edit_type_rule_matched": True}, True),
        ({"edit_type_rule_matched": False}, False),
        ({"edit_type_rule_fallback": False}, True),
        ({"edit_type_rule_fallback": True}, False),
        ({"edit_type_source": "rules"}, True),
        ({"edit_type_source": "column"}, True),
        ({"edit_type_source": "default"}, False),
    ],
)
def test_edit_type_trust_normalises_every_builders_flag(tmp_path, extra, expected):
    s = _sample(tmp_path, "x1", source="S", edit_type="add", extra=extra)
    assert edit_type_is_trusted(s) is expected


def test_combined_records_carry_the_normalised_flag(tmp_path):
    stats = combine(_fixture(tmp_path), root=tmp_path)
    out = read_jsonl(tmp_path / "out_s.jsonl", SampleRecord)
    assert len(out) == 16
    flags = [(s.metadata.model_extra or {}).get("edit_type_trusted") for s in out]
    assert all(f is not None for f in flags)
    assert sum(bool(f) for f in flags) == 12  # 4 strata x 1 default-labelled row
    assert stats.n_trusted == 12


# --------------------------------------------------------------------------- #
# the superset property — the reason the draw is shuffle-then-prefix           #
# --------------------------------------------------------------------------- #
def test_raising_the_subset_size_yields_a_superset(tmp_path):
    small = tmp_path / "small"
    big = tmp_path / "big"
    for d in (small, big):
        d.mkdir()
    combine(_fixture(small, k=1, pair_k=1), root=small)
    combine(_fixture(big, k=3, pair_k=2), root=big)

    small_s = {s.sample_id for s in read_jsonl(small / "sub_s.jsonl", SampleRecord)}
    big_s = {s.sample_id for s in read_jsonl(big / "sub_s.jsonl", SampleRecord)}
    assert small_s and small_s < big_s, "growing k must not drop an already-drawn sample"

    small_p = {p.pair_id for p in read_jsonl(small / "sub_p.jsonl", PairRecord)}
    big_p = {p.pair_id for p in read_jsonl(big / "sub_p.jsonl", PairRecord)}
    assert small_p and small_p <= big_p


def test_the_draw_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    combine(_fixture(a), root=a)
    combine(_fixture(b), root=b)
    ids_a = [s.sample_id for s in read_jsonl(a / "sub_s.jsonl", SampleRecord)]
    ids_b = [s.sample_id for s in read_jsonl(b / "sub_s.jsonl", SampleRecord)]
    assert ids_a == ids_b


# --------------------------------------------------------------------------- #
# subset composition                                                          #
# --------------------------------------------------------------------------- #
def test_untrusted_edit_types_are_excluded_from_the_scoring_subset(tmp_path):
    combine(_fixture(tmp_path, k=4), root=tmp_path)
    sub = read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)
    # k=4 asks for every row in each stratum, but the default-labelled one is
    # filtered out first, so 3 per stratum is the ceiling.
    assert len(sub) == 12
    assert all(edit_type_is_trusted(s) for s in sub)


def test_untrusted_rows_are_kept_when_the_filter_is_off(tmp_path):
    combine(_fixture(tmp_path, k=4, require_trusted=False), root=tmp_path)
    assert len(read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)) == 16


def test_pairs_are_drawn_independently_of_the_scoring_subset(tmp_path):
    """PairRecord is self-contained, so a pair need not have both sides sampled."""
    combine(_fixture(tmp_path, k=1, pair_k=2), root=tmp_path)
    sub_s = {s.sample_id for s in read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)}
    sub_p = read_jsonl(tmp_path / "sub_p.jsonl", PairRecord)
    assert len(sub_p) == 8  # 2 sources x 2 edit_types x 2
    assert any(p.sample_id_a not in sub_s for p in sub_p)


def test_human_labelled_pairs_are_consumed_first(tmp_path):
    combine(_fixture(tmp_path, pair_k=1), root=tmp_path)
    sub_p = read_jsonl(tmp_path / "sub_p.jsonl", PairRecord)
    assert len(sub_p) == 4
    assert all(p.ground_truth_preference is not None for p in sub_p), (
        "a pair carrying a real human preference must outrank one without"
    )


def test_pairs_can_be_restricted_to_human_labelled_only(tmp_path):
    combine(_fixture(tmp_path, pair_k=5, pairs_gt_only=True), root=tmp_path)
    sub_p = read_jsonl(tmp_path / "sub_p.jsonl", PairRecord)
    assert len(sub_p) == 4 and all(p.ground_truth_preference for p in sub_p)


def test_underfilled_strata_are_reported_not_silently_truncated(tmp_path):
    stats = combine(_fixture(tmp_path, k=99), root=tmp_path)
    assert len(stats.subset_strata_short) == 4
    assert "UNDERFILLED" in stats.summary()


# --------------------------------------------------------------------------- #
# invariants / failure modes                                                  #
# --------------------------------------------------------------------------- #
def test_duplicate_sample_ids_across_sources_are_reported(tmp_path):
    cfg_path = _fixture(tmp_path)
    cfg = load_config(cfg_path)
    # Point both sources at the same file so every id collides.
    cfg["sources"][1]["samples"] = "s1.jsonl"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    stats = combine(cfg_path, root=tmp_path)
    assert len(stats.duplicate_sample_ids) == 8
    assert "duplicate sample_id" in stats.summary()


def test_pair_referencing_a_missing_sample_is_reported(tmp_path):
    cfg_path = _fixture(tmp_path)
    pairs = read_jsonl(tmp_path / "p1.jsonl", PairRecord)
    pairs[0].sample_id_a = "does_not_exist"
    write_jsonl(tmp_path / "p1.jsonl", pairs)
    stats = combine(cfg_path, root=tmp_path)
    assert stats.dangling_pair_refs == [f"{pairs[0].pair_id}->does_not_exist"]
    assert "pair refs missing" in stats.summary()


def test_dry_run_writes_nothing(tmp_path):
    stats = combine(_fixture(tmp_path), root=tmp_path, dry_run=True)
    assert stats.n_samples == 16
    for name in ("out_s.jsonl", "out_p.jsonl", "sub_s.jsonl", "sub_p.jsonl"):
        assert not (tmp_path / name).exists()


def test_a_source_without_pairs_is_allowed(tmp_path):
    """MagicBrush has a single edit_model, so it contributes samples only."""
    cfg_path = _fixture(tmp_path)
    cfg = load_config(cfg_path)
    cfg["sources"][1].pop("pairs")
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    stats = combine(cfg_path, root=tmp_path)
    assert stats.per_source_samples["SrcB"] == 8
    assert "SrcB" not in stats.per_source_pairs
    assert not stats.dangling_pair_refs


def test_stats_track_human_labels(tmp_path):
    stats = combine(_fixture(tmp_path), root=tmp_path)
    assert stats.n_pairs == 8
    assert stats.n_pairs_with_gt == 4


# --------------------------------------------------------------------------- #
# human-anchor blocks (claim B)                                                #
# --------------------------------------------------------------------------- #
def _turn_sample(tmp_path, sid, *, source, edit_type, turn, model, human=None):
    rec = _sample(tmp_path, sid, source=source, edit_type=edit_type,
                  extra={"edit_type_source": "rules"})
    return rec.model_copy(update={
        "original_image_path": Path(f"img/turn_{turn}.png"),
        "instruction": f"instruction for turn {turn}",
        "edit_model": model,
        "human_score": human,
    })


def _anchor_fixture(tmp_path, *, blocks, k=1, rated_per_turn=2, editors_per_turn=3):
    """One source, `turn`s of `editors_per_turn` editors of which only
    `rated_per_turn` carry a human score — ImagenHub's real shape."""
    samples = []
    for et in ("add", "remove"):
        for turn in range(3):
            for m in range(editors_per_turn):
                samples.append(_turn_sample(
                    tmp_path, f"{et}_{turn}_{m}", source="SrcA", edit_type=et,
                    turn=f"{et}{turn}", model=f"m{m}",
                    human=0.5 + 0.1 * m if m < rated_per_turn else None,
                ))
    write_jsonl(tmp_path / "s1.jsonl", samples)
    cfg = {
        "sources": [{"name": "SrcA", "samples": "s1.jsonl"}],
        "output": {"samples": "out_s.jsonl"},
        "judge_subset": {
            "seed": 42,
            "per_source_per_edit_type": k,
            "human_anchor_blocks": blocks,
            "samples": "sub_s.jsonl",
        },
    }
    cfg_path = tmp_path / "cfg_anchor.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_anchor_block_drops_unrated_editors_not_the_whole_turn(tmp_path):
    """Regression: requiring *every* member of a turn to be rated rejected all 160
    of ImagenHub's rated turns (it ran 12 editors per item and rated 8), so the
    block came out empty."""
    cfg = _anchor_fixture(tmp_path, blocks=[
        {"source": "SrcA", "turns_per_edit_type": 2, "models_per_turn": 8,
         "require_human_score": True}
    ])
    stats = combine(cfg, root=tmp_path)
    sub = read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)
    anchor = [s for s in sub if s.metadata.model_extra.get("anchor_source") == "SrcA"]
    assert stats.hblock_turns == 4                      # 2 edit_types x 2 turns
    assert len(anchor) == 8                             # only the 2 rated editors
    assert all(s.human_score is not None for s in anchor)


def test_two_anchor_blocks_are_labelled_by_source(tmp_path):
    samples = []
    for source in ("SrcA", "SrcB"):
        for turn in range(2):
            for m in range(2):
                samples.append(_turn_sample(
                    tmp_path, f"{source}_{turn}_{m}", source=source, edit_type="add",
                    turn=f"{source}{turn}", model=f"m{m}", human=0.4 + 0.1 * m,
                ))
    write_jsonl(tmp_path / "s1.jsonl", [s for s in samples if s.source_dataset == "SrcA"])
    write_jsonl(tmp_path / "s2.jsonl", [s for s in samples if s.source_dataset == "SrcB"])
    cfg = {
        "sources": [{"name": "SrcA", "samples": "s1.jsonl"},
                    {"name": "SrcB", "samples": "s2.jsonl"}],
        "output": {"samples": "out_s.jsonl"},
        "judge_subset": {
            "seed": 42, "per_source_per_edit_type": 0,
            "human_anchor_blocks": [
                {"source": "SrcA", "turns_per_edit_type": 2, "models_per_turn": 2},
                {"source": "SrcB", "turns_per_edit_type": 2, "models_per_turn": 2},
            ],
            "samples": "sub_s.jsonl",
        },
    }
    (tmp_path / "cfg2.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    stats = combine(tmp_path / "cfg2.yaml", root=tmp_path)
    sub = read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)
    by_anchor = {}
    for s in sub:
        by_anchor.setdefault(s.metadata.model_extra.get("anchor_source"), []).append(s)
    assert set(by_anchor) == {"SrcA", "SrcB"}
    assert len(by_anchor["SrcA"]) == len(by_anchor["SrcB"]) == 4
    assert stats.hblock_by_source == {"SrcA": 4, "SrcB": 4}


def test_a_sample_both_blocks_picked_stays_in_breadth(tmp_path):
    """`subset_block` records which mechanism drew a sample. If an anchor could
    relabel a stratified pick, adding an anchor would silently delete balanced
    samples from claim A (measured on the real pool: 30 of them)."""
    cfg = _anchor_fixture(tmp_path, k=99, blocks=[
        {"source": "SrcA", "turns_per_edit_type": 3, "models_per_turn": 8}
    ])
    combine(cfg, root=tmp_path)
    sub = read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)
    rated = [s for s in sub if s.human_score is not None]
    assert rated, "the stratified draw takes everything, so the anchor overlaps it"
    assert all(s.metadata.model_extra["subset_block"] == "breadth" for s in rated)
    assert all(s.metadata.model_extra["anchor_source"] == "SrcA" for s in rated)


# --------------------------------------------------------------------------- #
# pair decisiveness tier                                                       #
# --------------------------------------------------------------------------- #
def test_decisive_tier_is_relative_to_each_sources_own_scale(tmp_path):
    """A fixed |gap| threshold silently favoured whichever source had the widest
    human scale: EBench's rescaled MOS spans ~0.5 and ImagenHub's spans 1.0, so
    |gap| > 0.2 called 94.6% of EBench's pairs indecisive and none of ImagenHub's."""
    narrow = [_sample(tmp_path, f"n{i}", source="Narrow", edit_type="add").model_copy(
        update={"human_score": 0.50 + 0.01 * i}) for i in range(10)]
    wide = [_sample(tmp_path, f"w{i}", source="Wide", edit_type="add").model_copy(
        update={"human_score": 0.05 * i}) for i in range(10)]
    pairs = [
        _pair(tmp_path, "pn", "n0", "n1", source="Narrow", edit_type="add", gt="a"),
        _pair(tmp_path, "pw", "w0", "w1", source="Wide", edit_type="add", gt="a"),
    ]
    # Same absolute gap on both sources; only the narrow source's is a real effect.
    pairs = [p.model_copy(update={"pair_quality_gap": 0.05}) for p in pairs]
    tier = decisive_tier(narrow + wide, pairs)
    assert tier(pairs[0]) == 0, "0.05 is ~1.7 SD on the narrow scale"
    assert tier(pairs[1]) == 1, "0.05 is ~0.3 SD on the wide scale"


def test_decisive_tier_ranks_unlabelled_pairs_last(tmp_path):
    samples = [_sample(tmp_path, f"s{i}", source="S", edit_type="add").model_copy(
        update={"human_score": 0.1 * i}) for i in range(10)]
    decisive = _pair(tmp_path, "p0", "s0", "s1", source="S", edit_type="add", gt="a")
    decisive = decisive.model_copy(update={"pair_quality_gap": 0.9})
    close = _pair(tmp_path, "p1", "s0", "s1", source="S", edit_type="add", gt="a")
    close = close.model_copy(update={"pair_quality_gap": 0.0})
    tied = _pair(tmp_path, "p2", "s0", "s1", source="S", edit_type="add", gt="tie")
    tied = tied.model_copy(update={"pair_quality_gap": 0.0})
    unlabelled = _pair(tmp_path, "p3", "s0", "s1", source="S", edit_type="add")
    tier = decisive_tier(samples, [decisive, close, tied, unlabelled])
    assert [tier(p) for p in (decisive, close, tied, unlabelled)] == [0, 1, 1, 2]
