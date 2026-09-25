"""Tests for the GenAI-Bench (human-voted A/B) ingestion builder.

No network: a synthetic parquet shard with the verified GenAI-Bench schema
(`struct<bytes, path>` image columns) is written to tmp_path and fed through
`build()` with an explicit `hf.files` list, which short-circuits the HF download.
"""

from __future__ import annotations

import io as _io
from pathlib import Path

import pytest
import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.build_genaibench import (
    DEFAULT_VOTE_MAP,
    GenAIBenchBuildStats,
    aggregate_preference,
    battle_key,
    build,
    content_stem,
    image_bytes,
    map_vote,
    plan_battles,
    select_balanced,
)
from edit_judge_bias.data.schema import EditType, PairRecord, SampleRecord
from edit_judge_bias.data.validate_manifest import validate_pairs, validate_samples

COLS = {
    "source_prompt": "source_prompt",
    "target_prompt": "target_prompt",
    "instruction": "instruct_prompt",
    "source_image": "source_image",
    "left_model": "left_model",
    "left_image": "left_output_image",
    "right_model": "right_model",
    "right_image": "right_output_image",
    "vote": "vote_type",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _png(color=(10, 20, 30), size=(8, 6)) -> bytes:
    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _img_cell(color, name) -> dict:
    """A cell shaped like HuggingFace's `Image` feature: struct<bytes, path>."""
    return {"bytes": _png(color), "path": name}


def _row(
    *,
    instruction,
    left_model,
    right_model,
    vote,
    source_prompt="a cat on a mat",
    target_prompt="a dog on a mat",
    src_color=(1, 2, 3),
    left_color=(40, 40, 40),
    right_color=(80, 80, 80),
) -> dict:
    return {
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "instruct_prompt": instruction,
        "source_image": _img_cell(src_color, "x_src.png"),
        "left_model": left_model,
        "left_output_image": _img_cell(left_color, "x_out.png"),
        "right_model": right_model,
        "right_output_image": _img_cell(right_color, "y_out.png"),
        "vote_type": vote,
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    img = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            ("source_prompt", pa.string()),
            ("target_prompt", pa.string()),
            ("instruct_prompt", pa.string()),
            ("source_image", img),
            ("left_model", pa.string()),
            ("left_output_image", img),
            ("right_model", pa.string()),
            ("right_output_image", img),
            ("vote_type", pa.string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _cfg(shard: Path, **overrides) -> dict:
    cfg = {
        "source_dataset": "GenAI-Bench",
        "id_prefix": "gb",
        "hf": {
            "repo_id": "TIGER-Lab/GenAI-Bench",
            "files": [str(shard)],
            "plan_over_http": False,
        },
        "columns": dict(COLS),
        "vote_map": {
            "leftvote": "a",
            "rightvote": "b",
            "tievote": "tie",
            "bothbad_vote": None,
        },
        "default_edit_type": "add",
        "sampling": {"seed": 42, "per_edit_type": None, "max_total": None},
        "images": {
            "original_dir": "data/images/original/genaibench",
            "edited_dir": "data/images/edited/genaibench",
        },
        "output": {
            "samples": "data/manifests/samples_genaibench.jsonl",
            "pairs": "data/manifests/pairs_genaibench.jsonl",
        },
    }
    cfg.update(overrides)
    return cfg


def _cfg_file(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "genaibench.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _build(tmp_path: Path, rows: list[dict], **cfg_overrides):
    shard = tmp_path / "shard.parquet"
    _write_parquet(shard, rows)
    cfg = _cfg(shard, **cfg_overrides)
    return build(_cfg_file(tmp_path, cfg), root=tmp_path)


# --------------------------------------------------------------------------- #
# vote_type -> preference mapping (the scientifically load-bearing part)      #
# --------------------------------------------------------------------------- #
def test_vote_vocabulary_maps_exactly_the_four_observed_labels():
    assert DEFAULT_VOTE_MAP == {
        "leftvote": "a",
        "rightvote": "b",
        "tievote": "tie",
        "bothbad_vote": None,
    }


@pytest.mark.parametrize(
    "vote,expected",
    [
        ("leftvote", "a"),
        ("rightvote", "b"),
        ("tievote", "tie"),
        ("bothbad_vote", None),  # "both bad" is not an A-vs-B preference
        ("  leftvote ", "a"),  # whitespace tolerated
        ("weirdvote", None),  # unknown label is unusable, never guessed
        (None, None),
    ],
)
def test_map_vote(vote, expected):
    assert map_vote(vote) is expected or map_vote(vote) == expected


def test_left_is_a_and_right_is_b_end_to_end(tmp_path: Path):
    """Pins the a/b orientation: left_* -> a, right_* -> b, leftvote -> "a"."""
    rows = [
        _row(
            instruction="Add a hat to the man.",
            left_model="MagicBrush",
            right_model="Pix2PixZero",
            vote="leftvote",
            left_color=(11, 11, 11),
            right_color=(99, 99, 99),
        )
    ]
    samples, pairs, stats = _build(tmp_path, rows)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.edit_model_a == "MagicBrush"
    assert pair.edit_model_b == "Pix2PixZero"
    assert pair.ground_truth_preference == "a"  # leftvote == the LEFT model won
    assert pair.pair_quality_gap == pytest.approx(1.0)  # +1 => A preferred

    by_id = {s.sample_id: s for s in samples}
    assert by_id[pair.sample_id_a].edit_model == "MagicBrush"
    assert by_id[pair.sample_id_b].edit_model == "Pix2PixZero"
    # And the image paths follow the same orientation (a = left image bytes).
    assert pair.edited_image_a_path == by_id[pair.sample_id_a].edited_image_path
    assert pair.edited_image_b_path == by_id[pair.sample_id_b].edited_image_path
    assert "MagicBrush" in pair.edited_image_a_path.as_posix()
    assert "Pix2PixZero" in pair.edited_image_b_path.as_posix()


def test_rightvote_orientation_is_b(tmp_path: Path):
    rows = [
        _row(
            instruction="Remove the dog.",
            left_model="SDEdit",
            right_model="InfEdit",
            vote="rightvote",
        )
    ]
    _, pairs, _ = _build(tmp_path, rows)
    assert pairs[0].ground_truth_preference == "b"
    assert pairs[0].edit_model_b == "InfEdit"
    assert pairs[0].pair_quality_gap == pytest.approx(-1.0)


def test_bothbad_rows_are_dropped_and_counted(tmp_path: Path):
    rows = [
        _row(instruction="Add a hat.", left_model="A", right_model="B", vote="bothbad_vote"),
        _row(
            instruction="Remove the cat.",
            left_model="A",
            right_model="B",
            vote="leftvote",
            source_prompt="a cat",
            left_color=(5, 5, 5),
            right_color=(6, 6, 6),
        ),
    ]
    samples, pairs, stats = _build(tmp_path, rows)
    assert stats.n_rows == 2
    assert stats.n_unusable_vote == 1
    assert stats.vote_counts["bothbad_vote"] == 1
    assert len(pairs) == 1
    assert pairs[0].instruction == "Remove the cat."
    assert all(p.ground_truth_preference in {"a", "b", "tie"} for p in pairs)


def test_tievote_is_kept_as_tie(tmp_path: Path):
    rows = [_row(instruction="Add a hat.", left_model="A", right_model="B", vote="tievote")]
    _, pairs, _ = _build(tmp_path, rows)
    assert pairs[0].ground_truth_preference == "tie"
    assert pairs[0].pair_quality_gap == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Repeated battles -> aggregated verdict                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "votes,pref,margin",
    [
        (["a"], "a", 1.0),
        (["b"], "b", -1.0),
        (["tie"], "tie", 0.0),
        (["a", "a"], "a", 1.0),
        (["a", "b"], "tie", 0.0),  # deadlock -> tie
        (["a", "a", "b"], "a", 1 / 3),  # majority
        (["a", "b", "tie"], "tie", 0.0),
        (["a", "tie", "tie"], "tie", 1 / 3),  # ties outnumber -> tie
    ],
)
def test_aggregate_preference(votes, pref, margin):
    got_pref, got_margin = aggregate_preference(votes)
    assert got_pref == pref
    assert got_margin == pytest.approx(margin)


def test_duplicate_battle_rows_collapse_to_one_pair(tmp_path: Path):
    row = dict(
        instruction="Add a hat to the man.",
        left_model="MagicBrush",
        right_model="PNP",
    )
    rows = [_row(vote="leftvote", **row), _row(vote="leftvote", **row)]
    _, pairs, stats = _build(tmp_path, rows)
    assert stats.n_rows == 2
    assert stats.n_battles == 1
    assert stats.n_multi_vote_battles == 1
    assert stats.n_conflicting_vote_battles == 0
    assert len(pairs) == 1


def test_conflicting_votes_are_flagged_and_resolved(tmp_path: Path):
    row = dict(instruction="Add a hat.", left_model="A", right_model="B")
    rows = [_row(vote="leftvote", **row), _row(vote="rightvote", **row)]
    _, pairs, stats = _build(tmp_path, rows)
    assert stats.n_conflicting_vote_battles == 1
    assert pairs[0].ground_truth_preference == "tie"


# --------------------------------------------------------------------------- #
# id construction                                                             #
# --------------------------------------------------------------------------- #
def test_ids_are_content_addressed_and_stable(tmp_path: Path):
    rows = [_row(instruction="Add a hat.", left_model="Magic Brush", right_model="PNP", vote="leftvote")]
    samples, pairs, _ = _build(tmp_path, rows)
    left_stem = content_stem(_png((40, 40, 40)))
    right_stem = content_stem(_png((80, 80, 80)))

    assert pairs[0].pair_id == f"gbp_{left_stem}_{right_stem}"
    by_model = {s.edit_model: s for s in samples}
    assert by_model["Magic Brush"].sample_id.startswith(f"gb_{left_stem}_Magic-Brush_")
    assert by_model["PNP"].sample_id.startswith(f"gb_{right_stem}_PNP_")
    # Model name is slugged in both the id and the image directory.
    assert "/Magic-Brush/" in by_model["Magic Brush"].edited_image_path.as_posix()


def test_rebuild_is_deterministic_and_reuses_images(tmp_path: Path):
    rows = [
        _row(instruction="Add a hat.", left_model="A", right_model="B", vote="leftvote"),
        _row(
            instruction="Remove the cat.",
            left_model="A",
            right_model="B",
            vote="rightvote",
            source_prompt="a cat",
            src_color=(4, 4, 4),  # a distinct source, so nothing is reused
            left_color=(7, 7, 7),
            right_color=(9, 9, 9),
        ),
    ]
    s1, p1, st1 = _build(tmp_path, rows)
    s2, p2, st2 = _build(tmp_path, rows)
    assert [s.sample_id for s in s1] == [s.sample_id for s in s2]
    assert [p.pair_id for p in p1] == [p.pair_id for p in p2]
    assert st1.images_written == 6 and st1.images_reused == 0  # 2 src + 4 edited
    assert st2.images_written == 0 and st2.images_reused == st1.images_written  # resumable


def test_shared_source_image_is_written_once(tmp_path: Path):
    """Two battles over the same source image must not duplicate it on disk."""
    rows = [
        _row(instruction="Add a hat.", left_model="A", right_model="B", vote="leftvote"),
        _row(
            instruction="Add a hat.",
            left_model="C",
            right_model="D",
            vote="rightvote",
            left_color=(21, 21, 21),
            right_color=(22, 22, 22),
        ),
    ]
    samples, pairs, stats = _build(tmp_path, rows)
    assert len(pairs) == 2
    assert len({p.original_image_path.as_posix() for p in pairs}) == 1
    orig_dir = tmp_path / "data" / "images" / "original" / "genaibench"
    assert len(list(orig_dir.iterdir())) == 1
    # 4 distinct edited outputs + 1 source = 5 files written.
    assert stats.images_written == 5


# --------------------------------------------------------------------------- #
# image bytes decoding                                                        #
# --------------------------------------------------------------------------- #
def test_image_bytes_from_struct_shaped_dict():
    raw = _png((3, 4, 5))
    assert image_bytes({"bytes": raw, "path": "a.png"}) == raw
    assert image_bytes(raw) == raw
    assert image_bytes(bytearray(raw)) == raw
    assert image_bytes(None) is None
    assert image_bytes({"bytes": None, "path": None}) is None
    assert image_bytes({"bytes": b"", "path": "missing.png"}) is None


def test_image_bytes_falls_back_to_path(tmp_path: Path):
    p = tmp_path / "on_disk.png"
    p.write_bytes(_png((6, 6, 6)))
    assert image_bytes({"bytes": None, "path": str(p)}) == p.read_bytes()


# --------------------------------------------------------------------------- #
# edit_type / content_category derivation                                     #
# --------------------------------------------------------------------------- #
def test_edit_type_from_rules_and_fallback_counting(tmp_path: Path):
    rows = [
        # matches the `remove` rule
        _row(
            instruction="Remove the dog from the photo.",
            left_model="A",
            right_model="B",
            vote="leftvote",
            source_prompt="a dog",
        ),
        # An action/pose edit -- outside the 6-class taxonomy on purpose, so no
        # rule matches and default_edit_type kicks in, counted as a fallback.
        # See tests/test_edit_type_rules.py::test_action_and_pose_edits_stay_unclassified.
        _row(
            instruction="Let the woman smile.",
            left_model="A",
            right_model="B",
            vote="rightvote",
            source_prompt="a cup",
            left_color=(31, 31, 31),
            right_color=(32, 32, 32),
        ),
    ]
    samples, pairs, stats = _build(tmp_path, rows)
    by_instr = {p.instruction: p for p in pairs}
    assert by_instr["Remove the dog from the photo."].edit_type is EditType.REMOVE
    assert by_instr["Let the woman smile."].edit_type is EditType.ADD  # fallback

    assert stats.n_rule_matched == 1
    assert stats.n_edit_type_fallback == 1
    assert stats.rule_coverage == pytest.approx(0.5)
    assert stats.unclassified == ["Let the woman smile."]

    flags = {s.instruction: s.metadata.edit_type_rule_matched for s in samples}
    assert flags["Remove the dog from the photo."] is True
    assert flags["Let the woman smile."] is False


def test_unclassified_row_dropped_when_no_fallback_configured(tmp_path: Path):
    rows = [
        _row(
            instruction="Let the woman smile.",
            left_model="A",
            right_model="B",
            vote="leftvote",
        )
    ]
    samples, pairs, stats = _build(tmp_path, rows, default_edit_type=None)
    assert pairs == [] and samples == []
    assert stats.n_edit_type_fallback == 1
    assert stats.n_battles == 0
    assert any("unclassified edit_type" in f for f in stats.failures)


def test_content_category_uses_source_and_target_prompts(tmp_path: Path):
    """The instruction alone is terse; the captions carry the scene."""
    rows = [
        _row(
            instruction="Make it nicer.",  # no content cue at all
            source_prompt="a zebra grazing",
            target_prompt="a zebra grazing peacefully",
            left_model="A",
            right_model="B",
            vote="leftvote",
        )
    ]
    samples, _, _ = _build(tmp_path, rows)
    assert samples[0].content_category.value == "animal"


# --------------------------------------------------------------------------- #
# missing fields / failure logging                                            #
# --------------------------------------------------------------------------- #
def test_missing_instruction_is_skipped_not_fatal(tmp_path: Path):
    rows = [
        _row(instruction="", left_model="A", right_model="B", vote="leftvote"),
        _row(
            instruction="Add a hat.",
            left_model="A",
            right_model="B",
            vote="leftvote",
            left_color=(51, 51, 51),
            right_color=(52, 52, 52),
        ),
    ]
    _, pairs, stats = _build(tmp_path, rows)
    assert stats.n_missing_field == 1
    assert len(pairs) == 1


# --------------------------------------------------------------------------- #
# sampling / balancing                                                        #
# --------------------------------------------------------------------------- #
def _battle(key: str, edit_type: str) -> dict:
    return {"_key": (key, key, "A", "B"), "_edit_type": edit_type}


def test_select_balanced_caps_per_edit_type_and_counts_drops():
    battles = [_battle(f"add{i}", "add") for i in range(5)]
    battles += [_battle(f"rem{i}", "remove") for i in range(2)]
    stats = GenAIBenchBuildStats()
    cfg = {"sampling": {"seed": 42, "per_edit_type": 3}}
    sel = select_balanced(battles, cfg, stats)
    assert len(sel) == 5  # 3 add + 2 remove
    assert stats.dropped_by_balance["add"] == 2
    assert "remove" not in stats.dropped_by_balance
    # deterministic
    stats2 = GenAIBenchBuildStats()
    assert [b["_key"] for b in sel] == [
        b["_key"] for b in select_balanced(battles, cfg, stats2)
    ]


def test_select_balanced_max_total_is_logged():
    battles = [_battle(f"add{i}", "add") for i in range(10)]
    stats = GenAIBenchBuildStats()
    sel = select_balanced(battles, {"sampling": {"max_total": 4}}, stats)
    assert len(sel) == 4
    assert stats.dropped_by_balance["_max_total"] == 6


def test_larger_per_edit_type_is_a_superset():
    battles = [_battle(f"add{i}", "add") for i in range(10)]
    small = {b["_key"] for b in select_balanced(battles, {"sampling": {"per_edit_type": 3}}, GenAIBenchBuildStats())}
    large = {b["_key"] for b in select_balanced(battles, {"sampling": {"per_edit_type": 6}}, GenAIBenchBuildStats())}
    assert small <= large


# --------------------------------------------------------------------------- #
# manifests on disk                                                           #
# --------------------------------------------------------------------------- #
def test_manifests_are_written_and_paths_validate(tmp_path: Path):
    rows = [
        _row(instruction="Add a hat.", left_model="MagicBrush", right_model="PNP", vote="leftvote"),
        _row(
            instruction="Remove the cat.",
            left_model="SDEdit",
            right_model="InfEdit",
            vote="rightvote",
            source_prompt="a cat",
            src_color=(9, 9, 9),
            left_color=(61, 61, 61),
            right_color=(62, 62, 62),
        ),
    ]
    samples, pairs, _ = _build(tmp_path, rows)
    s_path = tmp_path / "data" / "manifests" / "samples_genaibench.jsonl"
    p_path = tmp_path / "data" / "manifests" / "pairs_genaibench.jsonl"
    assert s_path.exists() and p_path.exists()

    on_disk_samples = io.read_jsonl(s_path, SampleRecord)
    on_disk_pairs = io.read_jsonl(p_path, PairRecord)
    assert len(on_disk_samples) == len(samples) == 4
    assert len(on_disk_pairs) == len(pairs) == 2
    assert validate_samples(s_path, tmp_path).ok
    assert validate_pairs(p_path, tmp_path).ok

    # Manifest paths are root-relative POSIX.
    line = s_path.read_text(encoding="utf-8").splitlines()[0]
    assert "data/images/" in line and "\\\\" not in line
    assert all(not p.original_image_path.is_absolute() for p in on_disk_pairs)


def test_every_pair_sample_id_resolves_to_a_sample(tmp_path: Path):
    rows = [
        _row(instruction="Add a hat.", left_model="A", right_model="B", vote="leftvote"),
        _row(
            instruction="Add a hat.",
            left_model="A",
            right_model="C",
            vote="rightvote",
            right_color=(71, 71, 71),
        ),
    ]
    samples, pairs, _ = _build(tmp_path, rows)
    ids = {s.sample_id for s in samples}
    for p in pairs:
        assert p.sample_id_a in ids and p.sample_id_b in ids
    # The replayed left output (same bytes + model + instruction) is one sample.
    assert len(samples) == 3


def test_battle_key_ignores_target_prompt_and_vote():
    a = _row(instruction="Add a hat.", left_model="A", right_model="B", vote="leftvote")
    b = _row(
        instruction="Add a hat.",
        left_model="A",
        right_model="B",
        vote="rightvote",
        target_prompt="something else",
    )
    assert battle_key(a, COLS) == battle_key(b, COLS)


def test_plan_battles_reports_vote_counts(tmp_path: Path):
    shard = tmp_path / "shard.parquet"
    rows = [
        _row(instruction="Add a hat.", left_model="A", right_model="B", vote="leftvote"),
        _row(instruction="Remove it.", left_model="A", right_model="B", vote="bothbad_vote"),
        _row(instruction="Remove the cat.", left_model="A", right_model="B", vote="tievote"),
    ]
    _write_parquet(shard, rows)
    stats = GenAIBenchBuildStats()
    battles = plan_battles([shard], _cfg(shard), stats)
    assert stats.n_rows == 3
    assert dict(stats.vote_counts) == {"leftvote": 1, "bothbad_vote": 1, "tievote": 1}
    assert len(battles) == 2
    assert {b["preference"] for b in battles} == {"a", "tie"}
