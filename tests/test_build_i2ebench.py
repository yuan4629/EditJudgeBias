"""Tests for the I2EBench ingestion builder (Milestone 1)."""

from __future__ import annotations

from pathlib import Path

from conftest import make_mini_editbench, mini_config

from edit_judge_bias.data.build_i2ebench import (
    build,
    build_all_samples,
    select_pilot,
)
from edit_judge_bias.data.schema import ContentCategory, EditType


def _categories():
    return {
        "ObjectRemoval": [
            {"image": "0001.png", "ori_exp": "Remove the cat.", "type": "animal"},
            {"image": "0002.png", "ori_exp": "Remove the man.", "type": "human"},
            {"image": "0003.png", "ori_exp": "Remove the tree.", "type": "plant"},
        ],
        "ColorAlteration": [
            {"image": "0010.png", "ori_exp": "Make the car red.", "type": "object"},
            {"image": "0011.png", "ori_exp": "Make the sky blue.", "type": "scenery"},
        ],
    }


def test_build_maps_types_and_matches_by_stem(tmp_path: Path):
    make_mini_editbench(tmp_path, _categories(), edited_ext=".jpg")  # input .png
    cfg = mini_config()
    samples, stats = build_all_samples(cfg, root=tmp_path)

    # 3 ObjectRemoval + 2 ColorAlteration images, 2 models each = 10 samples.
    assert stats.n_samples == 10
    assert not stats.missing_input and not stats.missing_edited

    by_id = {s.sample_id: s for s in samples}
    rem = by_id["i2e_ObjectRemoval_0001_modelX"]
    assert rem.edit_type is EditType.REMOVE
    assert rem.content_category is ContentCategory.ANIMAL
    assert rem.instruction == "Remove the cat."
    # plant -> object (folded), and edited stem matched across .png->.jpg.
    assert by_id["i2e_ObjectRemoval_0003_modelX"].content_category is ContentCategory.OBJECT
    assert rem.edited_image_path.suffix == ".jpg"
    assert rem.original_image_path.suffix == ".png"


def test_missing_edited_is_logged_not_fatal(tmp_path: Path):
    make_mini_editbench(
        tmp_path,
        _categories(),
        models={"modelX": [], "modelY": ["0002.png"]},  # modelY missing one image
    )
    cfg = mini_config()
    samples, stats = build_all_samples(cfg, root=tmp_path)
    assert len(stats.missing_edited) == 1
    assert stats.n_samples == 9
    assert "i2e_ObjectRemoval_0002_modelY" not in {s.sample_id for s in samples}


def test_default_content_category_when_type_missing(tmp_path: Path):
    cats = {"ObjectRemoval": [{"image": "0001.png", "ori_exp": "Remove it."}]}
    make_mini_editbench(tmp_path, cats)
    cfg = mini_config()
    samples, _ = build_all_samples(cfg, root=tmp_path)
    assert samples[0].content_category is ContentCategory.GLOBAL


def test_excluded_categories_are_ignored(tmp_path: Path):
    cats = dict(_categories())
    cats["DirectionPerception"] = [{"image": "9.png", "ori_exp": "x", "type": "object"}]
    make_mini_editbench(tmp_path, cats)
    cfg = mini_config()  # map excludes DirectionPerception
    samples, stats = build_all_samples(cfg, root=tmp_path)
    assert stats.n_categories == 2
    assert all("DirectionPerception" not in s.sample_id for s in samples)


def test_select_pilot_is_balanced_and_deterministic(tmp_path: Path):
    make_mini_editbench(tmp_path, _categories())
    cfg = mini_config()  # groups_per_edit_type=2
    samples, _ = build_all_samples(cfg, root=tmp_path)

    pilot1 = select_pilot(samples, cfg)
    pilot2 = select_pilot(samples, cfg)
    assert [s.sample_id for s in pilot1] == [s.sample_id for s in pilot2]  # deterministic

    # 2 groups per edit_type; remove has 3 originals (pick 2), color has 2 (pick 2).
    et = {}
    for s in pilot1:
        et.setdefault(s.edit_type.value, set()).add(s.original_image_path.as_posix())
    assert len(et["remove"]) == 2
    assert len(et["color"]) == 2


def test_build_writes_both_manifests(tmp_path: Path):
    make_mini_editbench(tmp_path, _categories())
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(mini_config()), encoding="utf-8")
    samples, pilot, stats = build(cfg_path, root=tmp_path)
    assert (tmp_path / "data" / "manifests" / "samples.jsonl").exists()
    assert (tmp_path / "data" / "manifests" / "samples_pilot.jsonl").exists()
    assert len(samples) == 10


def test_dry_run_writes_nothing(tmp_path: Path):
    make_mini_editbench(tmp_path, _categories())
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(mini_config()), encoding="utf-8")
    build(cfg_path, root=tmp_path, dry_run=True)
    assert not (tmp_path / "data" / "manifests" / "samples.jsonl").exists()
