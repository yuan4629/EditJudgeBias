"""Tests for the EDITED-member extractor.

This step joins a built pool back to the parquet shards it came from, and both ways it can
fail are silent by default: an id no shard contains just goes missing (a short pool that still
looks complete), and an "edited" image that is byte-identical to its source is a non-edit that
would hand every judge an identical pair. Each has a test here.
"""

from __future__ import annotations

import importlib.util
import io as _io
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import ContentCategory, EditType, SampleRecord

pytest.importorskip("pyarrow")

_SPEC = importlib.util.spec_from_file_location(
    "fetch_edited_images",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_edited_images.py",
)
FE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FE)


def _png_bytes(colour) -> bytes:
    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(buf, format="PNG")
    return buf.getvalue()


def _write_shard(path: Path, rows) -> None:
    """rows: [(raw_id, src_bytes, edited_bytes)]"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "omni_edit_id": pa.array([r[0] for r in rows]),
        "src_img": pa.array([{"bytes": r[1], "path": ""} for r in rows]),
        "edited_img": pa.array([{"bytes": r[2], "path": ""} for r in rows]),
    })
    pq.write_table(table, path)


def _pool(root: Path, raw_ids, shard_name="s0.parquet"):
    from PIL import Image

    records = []
    for raw in raw_ids:
        sid = f"omniedit_train_{raw}"
        orig = root / "orig" / f"{sid}.jpg"
        orig.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (10, 20, 30)).save(orig)
        rec = SampleRecord(
            sample_id=sid, source_dataset="omniedit_train", edit_type=EditType.ADD,
            content_category=ContentCategory.HUMAN,
            original_image_path=f"orig/{sid}.jpg", instruction="Add a hat",
            edit_model="omniedit_train_probe",
            edited_image_path=f"edited/{sid}.jpg",
        )
        rec.metadata.source_file = f"tmp_data/downloads/{shard_name}"
        rec.metadata.edited_image_fetched = False
        records.append(rec)
    io.write_jsonl(root / "pool.jsonl", records)
    return records


def test_extracts_edited_members_and_flags_the_manifest(tmp_path: Path):
    _pool(tmp_path, ["a1", "a2"])
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [
        ("a1", _png_bytes((10, 20, 30)), _png_bytes((200, 40, 40))),
        ("a2", _png_bytes((10, 20, 30)), _png_bytes((40, 200, 40))),
    ])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert rep["written"] == 2
    assert rep["missing"] == 0
    assert (tmp_path / "edited" / "omniedit_train_a1.jpg").exists()
    reloaded = io.read_jsonl(tmp_path / "pool.jsonl", SampleRecord)
    assert all(r.metadata.edited_image_fetched is True for r in reloaded)


def test_a_missing_id_is_reported_and_the_manifest_is_not_blessed(tmp_path: Path):
    """A short pool must fail loudly.

    If the shard set or `--id-col` is wrong, the rows simply do not join and the pool silently
    shrinks -- while every record still claims an `edited_image_path`. Leaving
    `edited_image_fetched: false` is what stops the next stage from trusting it.
    """
    _pool(tmp_path, ["a1", "ghost"])
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [
        ("a1", _png_bytes((10, 20, 30)), _png_bytes((200, 40, 40))),
    ])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert rep["missing"] == 1
    assert "omniedit_train_ghost" in rep["missing_ids"]
    assert "manifest_updated" not in rep
    reloaded = io.read_jsonl(tmp_path / "pool.jsonl", SampleRecord)
    assert all(r.metadata.edited_image_fetched is False for r in reloaded)


def test_an_edited_image_identical_to_its_source_is_refused(tmp_path: Path):
    """A non-edit would hand every judge an identical pair and a meaningless score."""
    _pool(tmp_path, ["a1"])
    same = _png_bytes((10, 20, 30))
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [("a1", same, same)])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert rep["identical_to_original"] == 1
    assert rep.get("written", 0) == 0
    assert not (tmp_path / "edited" / "omniedit_train_a1.jpg").exists()


def test_already_fetched_images_are_skipped(tmp_path: Path):
    """The images themselves are the checkpoint -- a rerun must not redo the read."""
    _pool(tmp_path, ["a1"])
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [
        ("a1", _png_bytes((10, 20, 30)), _png_bytes((200, 40, 40))),
    ])
    first = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert first["written"] == 1
    second = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert second.get("written", 0) == 0
    assert second["shards_skipped_complete"] == 1
    assert second["missing"] == 0


def test_dry_run_reads_nothing(tmp_path: Path):
    _pool(tmp_path, ["a1", "a2"])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path, dry_run=True)
    assert rep["per_shard"] == {"s0.parquet": 2}
    assert "written" not in rep
    assert not (tmp_path / "edited").exists()


def test_a_windows_separator_source_file_still_finds_its_shard(tmp_path: Path):
    """Manifests written on Windows before 2026-08-01 carry `dir\\shard.parquet`.

    A POSIX `Path` treats that as ONE segment, so `.name` returns the whole string, every shard
    lookup misses, and the run reports the entire pool as missing and exits 1 -- while being
    silently correct on Windows. The grouping normalises rather than trusting the platform.
    """
    records = _pool(tmp_path, ["a1"])
    records[0].metadata.source_file = "tmp_data/downloads\\s0.parquet"
    io.write_jsonl(tmp_path / "pool.jsonl", records)
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [
        ("a1", _png_bytes((10, 20, 30)), _png_bytes((200, 40, 40))),
    ])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert rep["missing"] == 0
    assert rep["written"] == 1


def test_size_disagreement_is_counted_not_hidden(tmp_path: Path):
    """The injector resizes a mismatched edited member; the count has to be visible.

    8 of the 41 scenes in the published v3 pool had a 500x500 original against a 1024x1024
    edit. Resizing is the right call -- dropping them loses real data -- but it is a decision,
    so it is reported rather than absorbed.
    """
    from PIL import Image

    _pool(tmp_path, ["a1"])
    buf = _io.BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(buf, format="PNG")
    _write_shard(tmp_path / "tmp_data" / "downloads" / "s0.parquet", [
        ("a1", _png_bytes((10, 20, 30)), buf.getvalue()),
    ])
    rep = FE.fetch_edited(pool_path="pool.jsonl", root=tmp_path)
    assert rep["size_agreement"] == {"size_mismatch": 1}
