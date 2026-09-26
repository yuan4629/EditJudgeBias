"""Tests for the HF-parquet ingestion builder (full-version M1b).

Network-free by construction: every test either exercises a pure helper or builds a
tiny local parquet file with the same ``struct<bytes: binary, path: string>`` image
encoding HuggingFace's `Image` feature uses, so `build()` runs end to end offline
(`hf.shards` explicit + a monkeypatched download that returns the local file).

The de-duplication test is the load-bearing one: ImagenHub Text-Guided IE is derived
from MagicBrush dev, so a leaked overlap would silently invalidate the paper's
multi-source-generalization claim.
"""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import List

import pytest
import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.build_from_hf import (
    HFBuildStats,
    _dedupe,
    _image_bytes,
    build,
    derive_edit_type,
    key_columns,
    load_exclusion_keys,
    row_key,
    select_balanced,
)
from edit_judge_bias.data.schema import SampleRecord

REPO_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("pyarrow")


# --------------------------------------------------------------------------- #
# Fixtures: a miniature MagicBrush-shaped parquet shard                       #
# --------------------------------------------------------------------------- #
def _png_bytes(color=(10, 20, 30), size=(8, 6)) -> bytes:
    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


MINI_ROWS = [
    ("100081", 1, "Remove the cat from the sofa."),
    ("100081", 3, "add a dog to the picture"),
    ("200001", 1, "Change the background to a beach."),
    ("200001", 2, "make the car red"),
    # An action/pose edit: outside the 6-class taxonomy on purpose, so no rule
    # matches and the configured default kicks in. See tests/test_edit_type_rules.py
    # ::test_action_and_pose_edits_stay_unclassified.
    ("300002", 1, "Let the woman smile."),
    ("300002", 2, "put a bird on the fence"),
]


def make_mini_parquet(path: Path, rows=MINI_ROWS) -> Path:
    """Write a parquet shard with MagicBrush's exact column/struct layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    img_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "img_id": pa.array([r[0] for r in rows], pa.string()),
            "turn_index": pa.array([r[1] for r in rows], pa.int32()),
            "source_img": pa.array(
                [{"bytes": _png_bytes((1, 2, 3)), "path": "s.png"} for _ in rows],
                img_struct,
            ),
            "mask_img": pa.array(
                [{"bytes": _png_bytes((255, 255, 255)), "path": "m.png"} for _ in rows],
                img_struct,
            ),
            "instruction": pa.array([r[2] for r in rows], pa.string()),
            "target_img": pa.array(
                [{"bytes": _png_bytes((9, 9, 9)), "path": "t.png"} for _ in rows],
                img_struct,
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_exclusion_csv(path: Path, uids: List[str], *, header: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (["uid"] if header else []) + list(uids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def mini_config(**overrides) -> dict:
    cfg = {
        "source_dataset": "MagicBrush",
        "id_prefix": "mb",
        "edit_model": "human_annotator",
        "hf": {
            "repo_id": "osunlp/MagicBrush",
            "shards": ["data/dev-00000.parquet"],
            "cache_dir": "tmp_data/hf_cache",
            "plan_over_http": False,
        },
        "columns": {
            "sample_key": ["img_id", "turn_index"],
            "instruction": "instruction",
            "original_image": "source_img",
            "edited_image": "target_img",
            "mask_image": "mask_img",
        },
        "key_sep": "_",
        "edit_type_from": "rules",
        "content_category_from": "rules",
        "default_edit_type": "replace",
        "sampling": {"seed": 42, "per_edit_type": 100},
        "images": {
            "original_dir": "data/images/original/magicbrush",
            "edited_dir": "data/images/edited/magicbrush",
            "mask_dir": "data/images/mask/magicbrush",
        },
        "output": {"samples": "data/manifests/samples_magicbrush.jsonl"},
    }
    cfg.update(overrides)
    return cfg


def _offline_build(tmp_path: Path, cfg: dict, monkeypatch, **kwargs):
    """Run build() against a local parquet, with HF network calls stubbed out."""
    shard = make_mini_parquet(tmp_path / "shard" / "dev-00000.parquet")
    monkeypatch.setattr(
        "edit_judge_bias.data.build_from_hf.download_shards",
        lambda cfg, shards, root: [shard],
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return build(cfg_path, root=tmp_path, **kwargs)


# --------------------------------------------------------------------------- #
# Exclusion-list parser + the de-dup invariant                                #
# --------------------------------------------------------------------------- #
def test_exclusion_parser_recovers_img_id_and_turn(tmp_path: Path):
    write_exclusion_csv(
        tmp_path / "lookup.csv",
        ["sample_100081_1.jpg", "sample_100081_3.jpg", "sample_102171_1.jpg"],
    )
    cfg = mini_config(exclusion={"path": "lookup.csv", "format": "imagenhub_uid"})
    assert load_exclusion_keys(cfg, tmp_path) == {"100081_1", "100081_3", "102171_1"}


def test_exclusion_parser_tolerates_headerless_file_and_no_extension(tmp_path: Path):
    write_exclusion_csv(
        tmp_path / "lookup.csv", ["sample_1_1", "sample_2_7"], header=False
    )
    cfg = mini_config(exclusion={"path": "lookup.csv"})
    assert load_exclusion_keys(cfg, tmp_path) == {"1_1", "2_7"}


def test_exclusion_parser_honours_key_sep(tmp_path: Path):
    write_exclusion_csv(tmp_path / "lookup.csv", ["sample_100081_1.jpg"])
    cfg = mini_config(exclusion={"path": "lookup.csv"}, key_sep="::")
    assert load_exclusion_keys(cfg, tmp_path) == {"100081::1"}


def test_unparseable_uid_raises_rather_than_silently_dropping(tmp_path: Path):
    write_exclusion_csv(tmp_path / "lookup.csv", ["not_a_valid_uid.jpg"])
    cfg = mini_config(exclusion={"path": "lookup.csv"})
    with pytest.raises(ValueError, match="cannot parse exclusion uid"):
        load_exclusion_keys(cfg, tmp_path)


def test_missing_exclusion_file_raises(tmp_path: Path):
    cfg = mini_config(exclusion={"path": "nope.csv"})
    with pytest.raises(FileNotFoundError):
        load_exclusion_keys(cfg, tmp_path)


def test_no_exclusion_config_means_no_keys(tmp_path: Path):
    assert load_exclusion_keys(mini_config(), tmp_path) == set()


def test_excluded_pairs_never_appear_in_the_manifest(tmp_path: Path, monkeypatch):
    """THE de-dup invariant: excluded (img_id, turn) pairs are absent from output."""
    excluded_uids = ["sample_100081_1.jpg", "sample_200001_2.jpg"]
    write_exclusion_csv(tmp_path / "lookup.csv", excluded_uids)
    cfg = mini_config(exclusion={"path": "lookup.csv", "format": "imagenhub_uid"})

    samples, stats = _offline_build(tmp_path, cfg, monkeypatch)

    assert stats.n_rows == len(MINI_ROWS)
    assert stats.n_exclusion_keys == 2
    assert stats.n_excluded == 2
    assert len(samples) == len(MINI_ROWS) - 2

    got_keys = {s.metadata.model_extra["source_key"] for s in samples}
    assert got_keys & {"100081_1", "200001_2"} == set()
    # and nothing leaked via the id/path either
    for uid in excluded_uids:
        stem = Path(uid).stem.removeprefix("sample_")
        assert all(f"_{stem}_" not in s.sample_id for s in samples)


def test_build_without_exclusion_keeps_everything(tmp_path: Path, monkeypatch):
    samples, stats = _offline_build(tmp_path, mini_config(), monkeypatch)
    assert stats.n_excluded == 0
    assert len(samples) == len(MINI_ROWS)


# --------------------------------------------------------------------------- #
# HF struct<bytes,path> image decoding                                        #
# --------------------------------------------------------------------------- #
def test_image_bytes_decodes_hf_struct_cell():
    raw = _png_bytes()
    assert _image_bytes({"bytes": raw, "path": "x.png"}) == raw


def test_image_bytes_accepts_plain_bytes_and_rejects_empty():
    raw = _png_bytes()
    assert _image_bytes(raw) == raw
    assert _image_bytes(bytearray(raw)) == raw
    assert _image_bytes(None) is None
    assert _image_bytes({"path": "x.png"}) is None
    assert _image_bytes({"bytes": b"", "path": "x.png"}) is None
    assert _image_bytes("not-an-image-cell") is None


def test_decoded_images_land_on_disk_with_masks(tmp_path: Path, monkeypatch):
    samples, stats = _offline_build(tmp_path, mini_config(), monkeypatch)
    assert stats.n_masks == len(MINI_ROWS)
    for s in samples:
        assert (tmp_path / s.original_image_path).is_file()
        assert (tmp_path / s.edited_image_path).is_file()
        assert s.metadata.has_mask and s.metadata.mask_path is not None
        assert (tmp_path / s.metadata.mask_path).is_file()
        assert (s.metadata.original_width, s.metadata.original_height) == (8, 6)
    # written verbatim: bytes on disk decode to the PNG we put in the shard
    assert (tmp_path / samples[0].original_image_path).suffix == ".png"


def test_rebuild_reuses_existing_images(tmp_path: Path, monkeypatch):
    cfg = mini_config()
    _offline_build(tmp_path, cfg, monkeypatch)
    _, stats2 = _offline_build(tmp_path, cfg, monkeypatch)
    assert stats2.images_written == 0
    assert stats2.images_reused == 3 * len(MINI_ROWS)  # original + edited + mask


def test_dedupe_is_order_preserving():
    assert _dedupe(["a", "b", "a", None, "c", "b"]) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Composite keys                                                              #
# --------------------------------------------------------------------------- #
def test_composite_and_scalar_sample_keys():
    assert key_columns(mini_config()) == ["img_id", "turn_index"]
    scalar = mini_config()
    scalar["columns"]["sample_key"] = "uid"
    assert key_columns(scalar) == ["uid"]

    row = {"img_id": "100081", "turn_index": 1}
    assert row_key(row, ["img_id", "turn_index"], "_") == "100081_1"
    assert row_key({"img_id": "x", "turn_index": None}, ["img_id", "turn_index"], "_") is None
    assert row_key({"img_id": "", "turn_index": 1}, ["img_id", "turn_index"], "_") is None


# --------------------------------------------------------------------------- #
# Rules-based edit_type + counted fallback                                    #
# --------------------------------------------------------------------------- #
def test_rules_edit_type_with_counted_default():
    cfg = mini_config()
    stats = HFBuildStats()
    assert derive_edit_type({"instruction": "Remove the cat."}, cfg, stats) == (
        "remove",
        "rules",
    )
    assert derive_edit_type({"instruction": "add a dog"}, cfg, stats) == ("add", "rules")
    # unclassifiable -> the configured default, and the fallback is counted
    assert derive_edit_type({"instruction": "Let the woman smile."}, cfg, stats) == (
        "replace",
        "default",
    )
    assert stats.n_edit_type_rules == 2
    assert stats.n_edit_type_default == 1
    assert stats.unclassified_instructions == ["Let the woman smile."]
    assert stats.rule_coverage == pytest.approx(2 / 3)


def test_rules_without_default_drops_the_row():
    cfg = mini_config()
    cfg.pop("default_edit_type")
    stats = HFBuildStats()
    et, src = derive_edit_type({"instruction": "Let the woman smile."}, cfg, stats)
    assert et is None
    assert stats.n_edit_type_default == 0
    assert len(stats.unclassified_instructions) == 1


def test_column_mode_still_works_and_reports_unmapped():
    cfg = mini_config(edit_type_from="column", edit_type_map={"ObjectRemoval": "remove"})
    cfg["columns"]["edit_type"] = "task"
    cfg.pop("default_edit_type")
    stats = HFBuildStats()
    assert derive_edit_type({"task": "ObjectRemoval"}, cfg, stats) == ("remove", "column")
    assert derive_edit_type({"task": "Mystery"}, cfg, stats) == (None, "column")
    assert stats.skipped_unmapped_edit_type == {"Mystery": 1}


def test_unknown_edit_type_from_mode_raises():
    with pytest.raises(ValueError, match="unknown edit_type_from"):
        derive_edit_type({}, mini_config(edit_type_from="magic"), HFBuildStats())


def test_edit_type_source_is_recorded_on_every_record(tmp_path: Path, monkeypatch):
    samples, stats = _offline_build(tmp_path, mini_config(), monkeypatch)
    sources = {
        s.metadata.model_extra["source_key"]: s.metadata.model_extra["edit_type_source"]
        for s in samples
    }
    assert sources["100081_1"] == "rules"
    assert sources["300002_1"] == "default"  # "Let the woman smile."
    assert stats.n_edit_type_default == 1


def test_content_category_from_rules(tmp_path: Path, monkeypatch):
    samples, _ = _offline_build(tmp_path, mini_config(), monkeypatch)
    by_key = {s.metadata.model_extra["source_key"]: s for s in samples}
    assert by_key["100081_1"].content_category.value == "animal"  # "the cat"
    assert by_key["200001_2"].content_category.value == "object"  # "the car"


# --------------------------------------------------------------------------- #
# Balanced sampling determinism + superset property                           #
# --------------------------------------------------------------------------- #
def _candidates(n_per_type: int = 12) -> List[dict]:
    return [
        {"_key": f"{et}_{i:03d}", "_edit_type": et}
        for et in ("add", "remove", "replace")
        for i in range(n_per_type)
    ]


def test_select_balanced_is_deterministic_for_a_seed():
    cands = _candidates()
    cfg = {"sampling": {"seed": 42, "per_edit_type": 5}}
    first = [r["_key"] for r in select_balanced(cands, cfg)]
    second = [r["_key"] for r in select_balanced(list(cands), cfg)]
    assert first == second
    assert len(first) == 15  # 3 types x 5

    other = [
        r["_key"] for r in select_balanced(cands, {"sampling": {"seed": 7, "per_edit_type": 5}})
    ]
    assert set(other) != set(first)  # the seed actually matters


def test_growing_per_edit_type_yields_a_superset():
    """Load-bearing: already-injected/judged samples must survive a cap increase."""
    cands = _candidates()
    small = {r["_key"] for r in select_balanced(cands, {"sampling": {"seed": 42, "per_edit_type": 4}})}
    big = {r["_key"] for r in select_balanced(cands, {"sampling": {"seed": 42, "per_edit_type": 9}})}
    assert small < big
    assert len(small) == 12 and len(big) == 27


def test_select_balanced_caps_only_oversized_buckets():
    cands = [{"_key": "a_1", "_edit_type": "add"}] + [
        {"_key": f"r_{i}", "_edit_type": "replace"} for i in range(10)
    ]
    got = select_balanced(cands, {"sampling": {"seed": 42, "per_edit_type": 3}})
    per_type = {}
    for r in got:
        per_type[r["_edit_type"]] = per_type.get(r["_edit_type"], 0) + 1
    assert per_type == {"add": 1, "replace": 3}


# --------------------------------------------------------------------------- #
# End-to-end manifest shape                                                   #
# --------------------------------------------------------------------------- #
def test_manifest_is_written_and_reloadable(tmp_path: Path, monkeypatch):
    samples, _ = _offline_build(tmp_path, mini_config(), monkeypatch)
    out = tmp_path / "data" / "manifests" / "samples_magicbrush.jsonl"
    assert out.is_file()
    reloaded = io.read_jsonl(out, SampleRecord)
    assert [s.sample_id for s in reloaded] == [s.sample_id for s in samples]
    assert all(s.source_dataset == "MagicBrush" for s in reloaded)
    assert all(s.edit_model == "human_annotator" for s in reloaded)
    assert all(s.sample_id.startswith("mb_") for s in reloaded)
    # the extra metadata keys must survive the JSONL round-trip: §9.4 filters on
    # edit_type_source to exclude fallback-guessed labels.
    assert all(
        s.metadata.model_extra["edit_type_source"] in {"rules", "column", "default"}
        for s in reloaded
    )
    assert {s.metadata.model_extra["source_key"] for s in reloaded} == {
        f"{i}_{t}" for i, t, _ in MINI_ROWS
    }


def test_limit_caps_the_build(tmp_path: Path, monkeypatch):
    samples, stats = _offline_build(tmp_path, mini_config(), monkeypatch, limit=2)
    assert stats.n_selected == 2
    assert len(samples) == 2


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch):
    samples, stats = _offline_build(tmp_path, mini_config(), monkeypatch, dry_run=True)
    assert samples == []
    assert stats.n_selected == len(MINI_ROWS)
    assert not (tmp_path / "data" / "manifests" / "samples_magicbrush.jsonl").exists()
    assert not (tmp_path / "data" / "images").exists()


def test_missing_column_raises_instead_of_silently_dropping(tmp_path: Path, monkeypatch):
    cfg = mini_config()
    cfg["columns"]["instruction"] = "prompt"  # typo'd column name
    with pytest.raises(KeyError, match="no column"):
        _offline_build(tmp_path, cfg, monkeypatch)


@pytest.mark.skipif(
    not (REPO_ROOT / "tmp_data" / "imagenhub_meta" / "dataset_lookup.csv").exists(),
    reason="needs ImagenHub's released metadata (tmp_data/imagenhub_meta/"
           "dataset_lookup.csv). Third-party corpus metadata is not redistributed with "
           "this repository; fetch it from the ImagenHub release (see docs/DATASETS.md).",
)
def test_real_magicbrush_config_parses_and_excludes_179(tmp_path: Path):
    """The shipped config's exclusion list is on disk and parses to 179 keys."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo_root / "configs" / "data" / "magicbrush_dev.yaml").read_text(encoding="utf-8")
    )
    keys = load_exclusion_keys(cfg, repo_root)
    assert len(keys) == 179
    assert "100081_1" in keys
    assert all("_" in k for k in keys)
    assert cfg["edit_model"] == "human_annotator"
    assert cfg["output"]["samples"] == "data/manifests/samples_magicbrush.jsonl"


# --------------------------------------------------------------------------- #
# ★ List-valued instruction columns                                            #
# --------------------------------------------------------------------------- #
def test_instruction_text_unwraps_a_list_column():
    """★ Some sources ship several paraphrases per edit instead of one string.

    OmniEdit's `edited_prompt_list` is `["Make it look like a cubist painting."]`. Passing that
    through `str()` embeds brackets and quotes into the instruction, and that text is part of
    the prompt the judge is shown -- it would corrupt every judgement and every rule-based
    edit_type classification downstream.
    """
    from edit_judge_bias.data.build_from_hf import instruction_text

    assert instruction_text(["Make it look like a cubist painting."]) == (
        "Make it look like a cubist painting."
    )
    assert "[" not in instruction_text(["Add a hat"])
    assert "'" not in instruction_text(["Add a hat"])


def test_instruction_text_takes_the_first_paraphrase_deterministically():
    """WHICH paraphrase is used must be fixed, or the manifest is not reproducible.

    Picking at random or by length would silently change the judged prompt between rebuilds of
    the same seed.
    """
    from edit_judge_bias.data.build_from_hf import instruction_text

    options = ["Replace the vampire with a woman", "Swap the vampire for a woman"]
    assert instruction_text(options) == options[0]
    assert instruction_text(options) == instruction_text(list(options))


def test_instruction_text_skips_blank_entries_and_handles_scalars():
    from edit_judge_bias.data.build_from_hf import instruction_text

    assert instruction_text(["", "  ", "Add a hat"]) == "Add a hat"
    assert instruction_text("Add a hat") == "Add a hat"
    assert instruction_text("  Add a hat  ") == "Add a hat"
    assert instruction_text(None) == ""
    assert instruction_text([]) == ""


def test_instruction_text_handles_numpy_style_sequences():
    """pyarrow list columns can arrive as ndarray, which is sequence-like but not a list."""
    np = pytest.importorskip("numpy")
    from edit_judge_bias.data.build_from_hf import instruction_text

    assert instruction_text(np.array(["Add a hat", "Put on a hat"])) == "Add a hat"
