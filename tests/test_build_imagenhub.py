"""Tests for the ImagenHub museum ingestion builder (full-version source B).

Everything here is offline: the museum file listing is faked via a tree-cache JSON
and the "downloads" are pre-created files on disk, so `build` runs with
`download=False` and never touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import touch_image

from edit_judge_bias.data.build_imagenhub import (
    aggregate_human_score,
    build,
    build_pair_records,
    build_samples,
    load_ratings,
    museum_ref,
    parse_rating_cell,
    parse_uid,
)
from edit_judge_bias.data.schema import ContentCategory, EditType

MODELS = ["MagicBrush", "SDEdit", "AURORA"]
RATED = ["MagicBrush", "SDEdit", "Imagic"]  # Imagic is rated but has no image dir

LOOKUP = {
    "sample_100081_1.jpg": {
        "instruction": "remove the frisbee",
        "source_global_caption": "Two people in an ocean playing with a yellow Frisbee.",
        "target_global_caption": "Two people in an ocean.",
    },
    "sample_100081_3.jpg": {
        "instruction": "add a dog barking near shore",
        "source_global_caption": "Two people and a whale in the ocean.",
        "target_global_caption": "Two people, a whale and a dog.",
    },
    "sample_222222_1.jpg": {
        # No rule matches this MagicBrush-style phrasing -> default_edit_type.
        "instruction": "let the woman cry",
        "source_global_caption": "A woman sits at a table.",
        "target_global_caption": "A crying woman sits at a table.",
    },
}


# --------------------------------------------------------------------------- #
# uid parsing                                                                 #
# --------------------------------------------------------------------------- #
def test_parse_uid_splits_img_id_and_turn():
    assert parse_uid("sample_100081_3.jpg") == ("100081", 3)
    assert parse_uid("sample_11091_1") == ("11091", 1)  # extension optional
    assert parse_uid("  sample_1_2.png  ") == ("1", 2)


@pytest.mark.parametrize("bad", ["", "sample_100081.jpg", "foo_1_2.jpg", "sample_a_1.jpg"])
def test_parse_uid_rejects_malformed(bad: str):
    with pytest.raises(ValueError):
        parse_uid(bad)


# --------------------------------------------------------------------------- #
# rating-cell parsing / aggregation                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cell,expected",
    [
        ("[0, 0.5]", (0.0, 0.5)),
        ("[0,0]", (0.0, 0.0)),  # no space — both spellings occur in the TSVs
        ("[1, 1]", (1.0, 1.0)),
        ("  [0.5,1]  ", (0.5, 1.0)),
        ([0, 1], (0.0, 1.0)),  # already a list
    ],
)
def test_parse_rating_cell_accepts_real_formats(cell, expected):
    assert parse_rating_cell(cell) == expected


@pytest.mark.parametrize(
    "cell",
    ["", "  ", "-", "nan", "[]", "[0]", "[0,0,0]", "garbage", "[0, 2]", None, "[a, b]"],
)
def test_parse_rating_cell_returns_none_instead_of_raising(cell):
    assert parse_rating_cell(cell) is None


def test_aggregate_human_score_means_over_raters_and_dimensions():
    # rater means: sc = (1 + 0.5 + 0)/3 = 0.5, pq = (1 + 1 + 0.5)/3 = 0.8333
    score, detail = aggregate_human_score([(1.0, 1.0), (0.5, 1.0), (0.0, 0.5)])
    assert detail["semantic_consistency"] == pytest.approx(0.5)
    assert detail["perceptual_quality"] == pytest.approx(0.8333, abs=1e-4)
    assert score == pytest.approx((0.5 + 0.8333) / 2, abs=1e-4)
    assert 0.0 <= score <= 1.0
    assert detail["n_raters"] == 3
    assert detail["rater_votes"] == [[1.0, 1.0], [0.5, 1.0], [0.0, 0.5]]
    assert detail["imagenhub_overall_geometric"] == pytest.approx(
        (0.5 * 0.8333) ** 0.5, abs=1e-3
    )


def test_aggregate_human_score_no_votes_is_none():
    assert aggregate_human_score([]) == (None, {})


def test_load_ratings_joins_three_raters_by_uid(tmp_path: Path):
    _write_raters(tmp_path)
    table = load_ratings([tmp_path / f"rater{i}.tsv" for i in (1, 2, 3)])
    assert set(table) == set(LOOKUP)
    assert table["sample_100081_1.jpg"]["MagicBrush"] == [(1.0, 1.0), (0.5, 1.0), (1.0, 0.5)]
    # A blank cell is "not rated", not "rated 0": only 2 raters survive.
    assert len(table["sample_100081_3.jpg"]["SDEdit"]) == 2
    # Models absent from the TSV never appear.
    assert "AURORA" not in table["sample_100081_1.jpg"]


def test_load_ratings_tolerates_missing_files(tmp_path: Path):
    assert load_ratings([tmp_path / "nope.tsv"]) == {}


# --------------------------------------------------------------------------- #
# fixtures: a miniature museum already "downloaded"                            #
# --------------------------------------------------------------------------- #
def _write_raters(root: Path) -> None:
    votes = {
        # uid -> per-rater cells for [MagicBrush, SDEdit, Imagic]
        "sample_100081_1.jpg": [["[1, 1]", "[0, 0]", "[0,0]"],
                                ["[0.5, 1]", "[0, 0.5]", "[0,0]"],
                                ["[1, 0.5]", "[0, 0]", "[0,0]"]],
        "sample_100081_3.jpg": [["[0, 0]", "[1, 1]", "[0,0]"],
                                ["[0, 0]", "", "[0,0]"],
                                ["[0, 0.5]", "[1, 1]", "[0,0]"]],
        "sample_222222_1.jpg": [["[0.5, 0.5]", "[0.5, 0.5]", "[0,0]"],
                                ["[0.5, 0.5]", "[0.5, 0.5]", "[0,0]"],
                                ["[0.5, 0.5]", "[0.5, 0.5]", "[0,0]"]],
    }
    for r in range(3):
        lines = ["\t".join(["uid"] + RATED)]
        for uid, per_rater in votes.items():
            lines.append("\t".join([uid] + per_rater[r]))
        (root / f"rater{r + 1}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_mini_museum(root: Path, *, models=MODELS, skip: set[str] | None = None) -> dict:
    """Lay out meta files + already-downloaded images, and return a matching config."""
    skip = skip or set()
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "dataset_lookup.json").write_text(json.dumps(LOOKUP), encoding="utf-8")
    _write_raters(meta)

    for uid in LOOKUP:
        touch_image(root / "data/images/original/imagenhub" / uid)
        for model in list(models) + ["GroundTruth"]:
            if f"{model}/{uid}" in skip:
                continue
            touch_image(root / "data/images/edited/imagenhub" / model / uid)

    return {
        "source_dataset": "ImagenHub",
        "museum": {
            "repo": "ChromAIca/ChromAIca.github.io",
            "branch": "main",
            "subtree": "Museum/ImagenHub_Text-Guided_IE",
            "input_dir": "input",
            "tree_cache": "tmp/tree.json",
        },
        "meta_dir": "meta",
        "lookup_json": "dataset_lookup.json",
        "rater_files": ["rater1.tsv", "rater2.tsv", "rater3.tsv"],
        "edit_models": list(models),
        "reference_model": "GroundTruth",
        "images": {
            "original_dir": "data/images/original/imagenhub",
            "edited_dir": "data/images/edited/imagenhub",
        },
        "sampling": {"limit": None, "seed": 42},
        "default_edit_type": "replace",
        "pairs": {"seed": 42, "max_pairs_per_group": 8, "tie_epsilon": 1.0e-9},
        "output": {
            "samples": "data/manifests/samples_imagenhub.jsonl",
            "pairs": "data/manifests/pairs_imagenhub.jsonl",
        },
    }


# --------------------------------------------------------------------------- #
# sample construction                                                          #
# --------------------------------------------------------------------------- #
def test_build_samples_offline(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)

    assert stats.n_uids == 3
    assert stats.n_samples == 9  # 3 uids x 3 models
    assert not stats.missing_input and not stats.missing_edited

    by_id = {s.sample_id: s for s in samples}
    rec = by_id["ih_100081_1_MagicBrush"]
    assert rec.source_dataset == "ImagenHub"
    assert rec.edit_type is EditType.REMOVE
    assert rec.content_category is ContentCategory.HUMAN  # "Two people ..."
    assert rec.instruction == "remove the frisbee"
    # Paths are root-relative POSIX.
    assert rec.original_image_path.as_posix() == (
        "data/images/original/imagenhub/sample_100081_1.jpg"
    )
    assert rec.edited_image_path.as_posix() == (
        "data/images/edited/imagenhub/MagicBrush/sample_100081_1.jpg"
    )
    assert rec.reference_image_path.as_posix().endswith("GroundTruth/sample_100081_1.jpg")
    assert rec.metadata.turn == 1
    assert rec.metadata.imagenhub_img_id == "100081"
    assert by_id["ih_100081_3_AURORA"].edit_type is EditType.ADD


def test_edit_type_fallback_is_counted(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)
    # "let the woman cry" matches no rule -> default_edit_type, counted once per uid.
    assert stats.n_edit_type_fallback == 1
    assert stats.n_classified == 2
    assert stats.rule_coverage == pytest.approx(2 / 3)
    assert stats.unclassified == ["let the woman cry"]
    fell_back = [s for s in samples if s.metadata.edit_type_rule_fallback]
    assert len(fell_back) == 3  # 3 models of the one unclassified uid
    assert all(s.edit_type is EditType.REPLACE for s in fell_back)


def test_human_score_only_on_rated_models(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)
    by_id = {s.sample_id: s for s in samples}

    mb = by_id["ih_100081_1_MagicBrush"]  # sc mean 0.8333, pq mean 0.8333
    assert mb.human_score == pytest.approx(0.8333, abs=1e-3)
    assert mb.metadata.n_raters == 3
    assert mb.metadata.semantic_consistency == pytest.approx(0.8333, abs=1e-3)

    sd = by_id["ih_100081_1_SDEdit"]  # sc 0, pq (0 + 0.5 + 0)/3
    assert sd.human_score == pytest.approx((0.0 + 1 / 6) / 2, abs=1e-4)

    # AURORA is not in the rater TSVs -> no human score at all.
    aur = by_id["ih_100081_1_AURORA"]
    assert aur.human_score is None
    assert not hasattr(aur.metadata, "n_raters")

    assert stats.n_human_scored == 6  # 3 uids x 2 rated models with images


def test_missing_edited_image_is_logged_not_fatal(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path, skip={"SDEdit/sample_100081_1.jpg"})
    samples, stats = build_samples(cfg, tmp_path, download=False)
    assert stats.n_samples == 8
    assert stats.missing_edited == ["SDEdit/sample_100081_1.jpg"]
    assert "ih_100081_1_SDEdit" not in {s.sample_id for s in samples}


def test_uid_csv_lookup_mismatch_is_fatal(tmp_path: Path):
    """A broken uid join would silently pair instructions with the wrong pixels."""
    cfg = _make_mini_museum(tmp_path)
    cfg["lookup_csv"] = "dataset_lookup.csv"
    (tmp_path / "meta" / "dataset_lookup.csv").write_text(
        "uid\nsample_100081_1.jpg\nsample_999999_9.jpg\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="absent from"):
        build_samples(cfg, tmp_path, download=False)


def test_sampling_limit_is_deterministic(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    cfg["sampling"]["limit"] = 2
    first, _ = build_samples(cfg, tmp_path, download=False)
    second, _ = build_samples(cfg, tmp_path, download=False)
    assert len({s.metadata.uid for s in first}) == 2
    assert [s.sample_id for s in first] == [s.sample_id for s in second]


# --------------------------------------------------------------------------- #
# pairs + ground-truth preference                                              #
# --------------------------------------------------------------------------- #
def test_pairs_carry_human_ground_truth(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)
    pairs = build_pair_records(samples, cfg, stats)

    # 3 groups x C(3,2) = 9 pairs, all in the ImagenHub id namespace.
    assert len(pairs) == 9
    assert stats.n_pairs == 9
    assert all(p.pair_id.startswith("pair_ih_") for p in pairs)
    assert len(set(p.pair_id for p in pairs)) == 9

    by_models = {(p.edit_model_a, p.edit_model_b): p for p in pairs
                 if p.sample_id_a.startswith("ih_100081_1")}
    # MagicBrush (0.833) vs SDEdit (0.083) — both rated, A clearly better.
    mb_sd = by_models[("MagicBrush", "SDEdit")]
    assert mb_sd.ground_truth_preference == "a"
    assert mb_sd.pair_quality_gap == pytest.approx(0.75, abs=1e-3)
    # AURORA is unrated -> no ground truth on any pair involving it.
    aur = by_models[("AURORA", "MagicBrush")]
    assert aur.ground_truth_preference is None
    assert aur.pair_quality_gap is None
    # 3 groups x 1 rated-vs-rated pair each.
    assert stats.n_pairs_with_preference == 3


def test_equal_human_scores_are_a_tie(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)
    pairs = build_pair_records(samples, cfg, stats)
    # uid sample_222222_1: every rater gave both models [0.5, 0.5].
    tie = [p for p in pairs if p.sample_id_a.startswith("ih_222222_1")
           and p.ground_truth_preference is not None]
    assert len(tie) == 1
    assert tie[0].ground_truth_preference == "tie"
    assert tie[0].pair_quality_gap == pytest.approx(0.0)


def test_preference_b_when_second_side_is_better(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    samples, stats = build_samples(cfg, tmp_path, download=False)
    pairs = build_pair_records(samples, cfg, stats)
    # uid sample_100081_3: SDEdit (1.0 / 0.75) beats MagicBrush (~0.083).
    p = next(
        p for p in pairs
        if p.sample_id_a == "ih_100081_3_MagicBrush" and p.edit_model_b == "SDEdit"
    )
    assert p.ground_truth_preference == "b"
    assert p.pair_quality_gap > 0


def test_pairs_are_capped_per_group(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    cfg["pairs"]["max_pairs_per_group"] = 2
    samples, stats = build_samples(cfg, tmp_path, download=False)
    pairs = build_pair_records(samples, cfg, stats)
    assert len(pairs) == 6  # 3 groups x 2


# --------------------------------------------------------------------------- #
# CLI-level build                                                              #
# --------------------------------------------------------------------------- #
def test_build_writes_both_manifests(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    samples, pairs, stats = build(cfg_path, root=tmp_path, download=False)
    assert (tmp_path / "data/manifests/samples_imagenhub.jsonl").is_file()
    assert (tmp_path / "data/manifests/pairs_imagenhub.jsonl").is_file()
    assert len(samples) == 9 and len(pairs) == 9

    from edit_judge_bias.data.validate_manifest import validate_pairs, validate_samples

    assert validate_samples(tmp_path / "data/manifests/samples_imagenhub.jsonl", tmp_path).ok
    assert validate_pairs(tmp_path / "data/manifests/pairs_imagenhub.jsonl", tmp_path).ok


def test_dry_run_writes_nothing(tmp_path: Path):
    cfg = _make_mini_museum(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    build(cfg_path, root=tmp_path, dry_run=True)
    assert not (tmp_path / "data/manifests/samples_imagenhub.jsonl").exists()


def test_museum_reads_the_pinned_commit():
    """The shipped config pins a commit; `branch` is only the fallback."""
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/data/imagenhub.yaml").read_text(encoding="utf-8"))
    assert museum_ref(cfg["museum"]) == "d5f553773f57bdb98cfe2ecf77312dd07c818fb6"
    assert museum_ref({"branch": "main"}) == "main"
    assert museum_ref({"revision": "abc", "branch": "main"}) == "abc"
