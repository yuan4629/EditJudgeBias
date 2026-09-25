"""EBench-18K ingestion tests.

The source is bought for one reason — it is the human anchor for claim B — so the
tests concentrate on the things that would quietly poison that anchor: a label
dimension going missing in the join, a task being force-labelled into an edit_type
it does not belong to, and the contaminated low-level tasks leaking back in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from edit_judge_bias.data import io
from edit_judge_bias.data.build_ebench import (
    MOS_DIMENSIONS,
    build,
    build_pairs,
    build_samples,
    load_config,
    parse_image_ref,
    parse_instruction,
    parse_qa,
    parse_score,
    task_index,
)
from edit_judge_bias.data.schema import SampleRecord

MODELS = ["model00", "model01"]
# (task index, instruction) for the synthetic fixture — one mapped high-level task,
# one unclassifiable one, one clean low-level task and one contaminated one.
TASKS = {
    "H_00": "Add a nose ring to the woman's nose",
    "H_06": "Let the woman cross her arms",
    "L_01": "Remove the noise points from the image",
    "L_00": "Make the blurry image clearer",
}


def _cfg(root: Path, **overrides) -> dict:
    cfg = {
        "source_dataset": "EBench-18K",
        "id_prefix": "ebench",
        "images_root": "tmp_data/editing_all",
        "meta_dir": "tmp_data/ebench_meta",
        "label_files": {
            "mos_quality": ["train_v.json"],
            "mos_alignment": ["train_e.json"],
            "mos_preservation": ["train_c.json"],
            "human_qa_pass": ["train_yn.json"],
        },
        "tasks": {
            "H_00": {"name": "add", "edit_type": "add"},
            "H_06": {"name": "action", "edit_type": None},
            "L_01": {"name": "denoise", "edit_type": "low-level"},
            "L_00": {"name": "deblur", "edit_type": "low-level", "contaminated": True},
        },
        "exclude_contaminated": True,
        "global_edit_types": ["low-level"],
        "output": {"samples_full": "data/manifests/samples_ebench18k.jsonl"},
    }
    cfg.update(overrides)
    return cfg


def _image(path: Path, colour: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (colour, colour, colour)).save(path)


@pytest.fixture()
def tree(tmp_path: Path):
    """Two images per task, two models, and the four label dimensions."""
    imgs = tmp_path / "tmp_data" / "editing_all"
    meta = tmp_path / "tmp_data" / "ebench_meta"
    meta.mkdir(parents=True, exist_ok=True)

    rows = {"mos_quality": [], "mos_alignment": [], "mos_preservation": [], "human_qa_pass": []}
    colour = 10
    for task, instruction in TASKS.items():
        level = "l" if task.startswith("L") else "h"
        for idx in range(2):
            stem = f"{task}_{idx:02d}"
            colour += 7
            for model in MODELS:
                _image(imgs / f"sourceimg_{level}" / model / f"{stem}.jpg", colour)
                _image(imgs / f"targetimg_{level}" / model / f"{stem}.jpg", colour + 3)
                ref = [f"/mnt/data/xzt/MM/editing_all/sourceimg_{level}/{model}/{stem}.jpg",
                       f"/mnt/data/xzt/MM/editing_all/targetimg_{level}/{model}/{stem}.jpg"]
                base = 40 + idx
                query = (f"With the image editing prompt [{instruction}], the source image "
                         "<image> is edited ... reply [The quality score is XX.XX]")
                rows["mos_quality"].append(
                    {"query": query, "response": f"The quality score is {base:.2f}", "images": ref})
                rows["mos_alignment"].append(
                    {"query": query, "response": f"The quality score is {base + 3:.2f}", "images": ref})
                rows["mos_preservation"].append(
                    {"query": query, "response": f"The quality score is {base + 6:.2f}", "images": ref})
                rows["human_qa_pass"].append(
                    {"query": query, "response": "Yes" if idx == 0 else "No", "images": ref})
    for dim, name in (("mos_quality", "train_v.json"), ("mos_alignment", "train_e.json"),
                      ("mos_preservation", "train_c.json"), ("human_qa_pass", "train_yn.json")):
        (meta / name).write_text(json.dumps(rows[dim]), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Parsers                                                                      #
# --------------------------------------------------------------------------- #
def test_instruction_comes_from_the_first_bracket_not_the_format_hint():
    query = ("With the image editing prompt [Add a hat], the source image <image> is edited. "
             "Just reply in the following format: [The quality score is XX.XX]")
    assert parse_instruction(query) == "Add a hat"


@pytest.mark.parametrize("response,expected", [
    ("The quality score is 46.35", 46.35),
    ("The quality score is 8", 8.0),
    ("no digits here", None),
])
def test_score_parsing(response, expected):
    assert parse_score(response) == expected


@pytest.mark.parametrize("response,expected", [("Yes", True), ("no.", False), ("maybe", None)])
def test_qa_parsing(response, expected):
    assert parse_qa(response) is expected


def test_image_ref_survives_the_authors_absolute_paths():
    assert parse_image_ref("/mnt/data/xzt/MM/editing_all/sourceimg_h/model10/H_11_12.jpg") \
        == ("h", "model10", "H_11_12")
    assert parse_image_ref("nothing/useful.png") is None


def test_task_index_drops_the_image_number():
    assert task_index("H_00_36") == "H_00"
    assert task_index("L_07_69") == "L_07"
    assert task_index("weird") is None


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def test_only_mapped_uncontaminated_tasks_become_samples(tree):
    samples, stats = build_samples(_cfg(tree), tree)
    # 2 tasks kept x 2 images x 2 models
    assert len(samples) == 8
    assert {s.edit_type.value for s in samples} == {"add", "low-level"}
    assert dict(stats.dropped_unmapped) == {"action": 4}
    assert dict(stats.dropped_contaminated) == {"deblur": 4}


def test_contaminated_tasks_can_be_re_enabled_for_an_appendix(tree):
    samples, stats = build_samples(_cfg(tree, exclude_contaminated=False), tree)
    assert len(samples) == 12
    assert not stats.dropped_contaminated
    assert any(s.metadata.model_extra["ebench_task"] == "deblur" for s in samples)


def test_unclassifiable_tasks_are_dropped_not_force_labelled(tree):
    """`action` has no honest target class — the same commitment as the rule tests."""
    samples, _ = build_samples(_cfg(tree), tree)
    assert not any(s.metadata.model_extra["ebench_task"] == "action" for s in samples)


def test_human_score_is_the_mean_and_each_dimension_survives(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    s = next(s for s in samples if s.sample_id == "ebench_H_00_00_model00")
    extra = s.metadata.model_extra
    assert (extra["mos_quality"], extra["mos_alignment"], extra["mos_preservation"]) == (40.0, 43.0, 46.0)
    assert s.human_score == pytest.approx(0.43)  # rescaled; see the 0-1 test below
    assert extra["human_score_kind"] == "human_rating"  # never confusable with a machine score
    assert extra["human_qa_pass"] is True


def test_originals_all_point_at_model00_since_the_17_copies_are_identical(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    for s in samples:
        assert "/model00/" in s.original_image_path.as_posix()
    # ...while the edited image keeps the sample's own model.
    s = next(s for s in samples if s.edit_model == "model01")
    assert "/targetimg_h/model01/" in s.edited_image_path.as_posix() \
        or "/targetimg_l/model01/" in s.edited_image_path.as_posix()


def test_low_level_is_global_and_high_level_uses_the_shared_classifier(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    low = [s for s in samples if s.edit_type.value == "low-level"]
    assert {s.content_category.value for s in low} == {"global"}
    high = next(s for s in samples if s.edit_type.value == "add")
    assert high.content_category.value == "human"  # "...the woman's nose"


def test_sample_ids_are_prefixed_and_unique(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    assert all(s.sample_id.startswith("ebench_") for s in samples)
    assert len({s.sample_id for s in samples}) == len(samples)


def test_byte_identical_originals_get_a_dup_group(tree):
    """EBench reuses photographs across tasks; the flag makes that visible."""
    imgs = tree / "tmp_data" / "editing_all"
    for model in MODELS:  # make L_01_01 the same photo as L_01_00
        _image(imgs / "sourceimg_l" / model / "L_01_01.jpg", 31)
        _image(imgs / "sourceimg_l" / model / "L_01_00.jpg", 31)
    samples, stats = build_samples(_cfg(tree), tree)
    assert stats.dup_groups == 1
    flagged = [s for s in samples if "dup_group" in s.metadata.model_extra]
    assert {s.sample_id for s in flagged} == {
        "ebench_L_01_00_model00", "ebench_L_01_00_model01",
        "ebench_L_01_01_model00", "ebench_L_01_01_model01"}


# --------------------------------------------------------------------------- #
# The two failure modes that must behave differently                           #
# --------------------------------------------------------------------------- #
def test_an_incomplete_label_join_is_fatal(tree, tmp_path: Path):
    """A missing MOS dimension puts a hole in the anchor itself -> refuse to build."""
    meta = tree / "tmp_data" / "ebench_meta"
    rows = json.loads((meta / "train_c.json").read_text(encoding="utf-8"))
    (meta / "train_c.json").write_text(json.dumps(rows[:-4]), encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tree)), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete join"):
        build(cfg_path, root=tree, dry_run=True)


def test_a_single_missing_image_is_fail_soft(tree):
    """One absent file must not abort a 10k-record batch — it is counted and skipped."""
    (tree / "tmp_data/editing_all/targetimg_h/model01/H_00_00.jpg").unlink()
    samples, stats = build_samples(_cfg(tree), tree)
    assert len(stats.missing_images) == 1
    assert len(samples) == 7
    assert "ebench_H_00_00_model01" not in {s.sample_id for s in samples}


def test_written_manifest_round_trips_and_is_path_validated(tree, tmp_path: Path):
    cfg = _cfg(tree)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    samples, _, _ = build(cfg_path, root=tree)
    out = tree / cfg["output"]["samples_full"]
    reloaded = io.read_jsonl(out, SampleRecord)
    assert len(reloaded) == len(samples) == 8
    assert all((tree / r.edited_image_path).is_file() for r in reloaded)


# --------------------------------------------------------------------------- #
# Pairs                                                                        #
# --------------------------------------------------------------------------- #
def test_every_pair_carries_a_human_ground_truth(tree):
    """Every sample has a MOS, so unlike other sources there is no unlabelled pair."""
    samples, _ = build_samples(_cfg(tree), tree)
    pairs = build_pairs(samples, _cfg(tree))
    assert pairs
    assert all(p.ground_truth_preference in {"a", "b", "tie"} for p in pairs)
    assert all(p.pair_quality_gap is not None for p in pairs)


def test_pair_ids_use_a_source_specific_namespace(tree):
    """The generic builder restarts at pair_000001, which is I2EBench's namespace."""
    samples, _ = build_samples(_cfg(tree), tree)
    pairs = build_pairs(samples, _cfg(tree))
    assert all(p.pair_id.startswith("pair_eb_") for p in pairs)
    assert len({p.pair_id for p in pairs}) == len(pairs)


def test_pairs_only_form_within_a_turn_and_across_editors(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    by_id = {s.sample_id: s for s in samples}
    for p in build_pairs(samples, _cfg(tree)):
        a, b = by_id[p.sample_id_a], by_id[p.sample_id_b]
        assert a.original_image_path == b.original_image_path
        assert a.instruction == b.instruction
        assert a.edit_model != b.edit_model


def test_preference_follows_the_mos_and_the_gap_is_absolute(tree):
    samples, _ = build_samples(_cfg(tree), tree)
    by_id = {s.sample_id: s for s in samples}
    for p in build_pairs(samples, _cfg(tree)):
        a, b = by_id[p.sample_id_a], by_id[p.sample_id_b]
        expected = "a" if a.human_score > b.human_score else (
            "b" if a.human_score < b.human_score else "tie")
        assert p.ground_truth_preference == expected
        assert p.pair_quality_gap == pytest.approx(abs(a.human_score - b.human_score))


def test_human_score_is_rescaled_to_the_pools_0_1_range(tree):
    """MOS is 0-100; ImagenHub's human_score is 0-1. Mixing them would make
    `pair_quality_gap` incomparable across sources."""
    samples, _ = build_samples(_cfg(tree), tree)
    s = next(s for s in samples if s.sample_id == "ebench_H_00_00_model00")
    assert s.human_score == pytest.approx(0.43)          # mean(40,43,46)/100
    assert all(0.0 <= s.human_score <= 1.0 for s in samples)
    # ...while the raw MOS stays available per dimension.
    assert s.metadata.model_extra["mos_quality"] == 40.0


# --------------------------------------------------------------------------- #
# The shipping config itself                                                   #
# --------------------------------------------------------------------------- #
def test_shipping_config_leaves_the_three_contaminated_tasks_flagged():
    """Measured against I2EBench: deblur/desnow/lowlight share upstream corpora."""
    cfg = load_config(Path("configs/data/ebench18k.yaml"))
    contaminated = {t["name"] for t in cfg["tasks"].values() if t.get("contaminated")}
    assert contaminated == {"deblur", "desnow", "lowlight"}
    assert cfg["exclude_contaminated"] is True


def test_shipping_config_declares_its_provenance_and_label_kind():
    cfg = load_config(Path("configs/data/ebench18k.yaml"))
    prov = cfg["provenance"]
    assert prov["label_kind"] == "human_rating"      # humans rated; not gold edits, not machine
    assert prov["overlap_report"] == "data/provenance/overlap_ebench18k.json"
    assert all(prov.get(k) for k in ("venue", "doi", "repo", "license"))
    assert "max_total" not in (cfg.get("sampling") or {})  # protocol §4.2


def test_shipping_config_leaves_committed_unclassified_tasks_unmapped():
    cfg = load_config(Path("configs/data/ebench18k.yaml"))
    unmapped = {t["name"] for t in cfg["tasks"].values() if t.get("edit_type") is None}
    assert {"action", "expression", "position"} <= unmapped
