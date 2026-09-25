"""Tests for the parquet -> candidate-originals ingest, the only path that feeds the D-S pool.

It had no test file until 2026-08-01 while carrying 31,598 rows of the fairness corpus. The
three things pinned here are the ones whose failure is invisible on disk: a manifest that
claims an edited image it never fetched, a re-run that doubles the manifest while the images
look untouched, and a Windows-separator `source_file` that makes the edited-member join miss
every shard on any POSIX machine.
"""

from __future__ import annotations

import importlib.util
import io as _io
import json
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import SampleRecord

pytest.importorskip("pyarrow")

_SPEC = importlib.util.spec_from_file_location(
    "fetch_probe_originals",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_probe_originals.py",
)
FP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FP)


def _png(colour=(10, 20, 30)) -> bytes:
    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(buf, format="PNG")
    return buf.getvalue()


def _shard(path: Path, rows, *, row_group_size: int = 100) -> None:
    """rows: [(raw_id, [instructions], src_bytes, task)]"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "omni_edit_id": pa.array([r[0] for r in rows]),
        "edited_prompt_list": pa.array([r[1] for r in rows]),
        "src_img": pa.array([{"bytes": r[2], "path": ""} for r in rows]),
        "task": pa.array([r[3] for r in rows]),
    })
    pq.write_table(table, path, row_group_size=row_group_size)


def _rows(n, prefix="task_obj_add", task="addition"):
    return [(f"{prefix}_{i}", [f"Add {i}", f"Add object {i} to the scene"], _png(), task)
            for i in range(n)]


def _fetch(tmp_path: Path, shards, **kw):
    return FP.fetch(label="omniedit_train", local_parquet=",".join(str(s) for s in shards),
                    row_groups=0, root=tmp_path, **kw)


# --------------------------------------------------------------------------- #
# ★ The manifest must not claim what it did not fetch                          #
# --------------------------------------------------------------------------- #
def test_records_are_stamped_probe_only_with_an_unfetched_edited_path(tmp_path: Path):
    """`edited_image_path` points at a file that does not exist -- the flags are the contract."""
    s = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    _shard(s, _rows(3))
    rep = FP.fetch(label="omniedit_train", local_parquet=str(s), row_groups=0, root=tmp_path)
    assert rep["originals_written"] == 3
    recs = io.read_jsonl(tmp_path / rep["out_manifest"], SampleRecord)
    assert len(recs) == 3
    for r in recs:
        assert r.metadata.probe_only is True
        assert r.metadata.edited_image_fetched is False
        assert r.metadata.raw_task == "addition"
        assert (tmp_path / r.original_image_path).exists()
        assert not (tmp_path / r.edited_image_path).exists()


def test_source_file_is_posix_so_the_edited_join_can_find_the_shard(tmp_path: Path):
    """`fetch_edited_images` groups by `Path(source_file).name`.

    A backslash string round-trips through a POSIX `Path` as ONE segment, so `.name` returns
    the whole `dir\\shard.parquet` and every shard lookup misses -- silent on Windows, and on
    Linux it reports the entire pool as missing.
    """
    s = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    _shard(s, _rows(2))
    FP.fetch(label="omniedit_train", local_parquet=str(s), row_groups=0, root=tmp_path)
    recs = io.read_jsonl(
        tmp_path / "data/manifests/samples_omniedit_train_probe.jsonl", SampleRecord)
    for r in recs:
        assert "\\" not in r.metadata.source_file
        assert Path(r.metadata.source_file).name == "train-00000-of-00571.parquet"


# --------------------------------------------------------------------------- #
# ★ A re-run must not double the manifest                                      #
# --------------------------------------------------------------------------- #
def test_multi_shard_append_is_idempotent(tmp_path: Path):
    """The images are skipped by an exists() check, so a duplicated manifest leaves no trace."""
    a = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    b = tmp_path / "tmp_data/downloads/train-00057-of-00571.parquet"
    _shard(a, _rows(3))
    _shard(b, _rows(2, prefix="task_obj_add_b"))
    first = _fetch(tmp_path, [a, b])
    assert first["originals_written"] == 5
    manifest = tmp_path / "data/manifests/samples_omniedit_train_probe.jsonl"
    assert len(io.read_jsonl(manifest, SampleRecord)) == 5

    second = _fetch(tmp_path, [a, b])
    assert second["manifest_rows_skipped_as_existing"] == 5
    recs = io.read_jsonl(manifest, SampleRecord)
    assert len(recs) == 5
    assert len({r.sample_id for r in recs}) == 5


def test_adding_a_shard_extends_rather_than_rewrites(tmp_path: Path):
    a = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    b = tmp_path / "tmp_data/downloads/train-00057-of-00571.parquet"
    _shard(a, _rows(3))
    _fetch(tmp_path, [a])
    _shard(b, _rows(2, prefix="task_obj_add_b"))
    rep = _fetch(tmp_path, [a, b])
    assert rep["manifest_rows_skipped_as_existing"] == 3
    recs = io.read_jsonl(
        tmp_path / "data/manifests/samples_omniedit_train_probe.jsonl", SampleRecord)
    assert len(recs) == 5
    assert len({r.metadata.source_file for r in recs}) == 2


# --------------------------------------------------------------------------- #
# ★ Row selection                                                              #
# --------------------------------------------------------------------------- #
def test_row_groups_zero_reads_every_group(tmp_path: Path):
    s = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    _shard(s, _rows(7), row_group_size=2)
    rep = FP.fetch(label="omniedit_train", local_parquet=str(s), row_groups=0, root=tmp_path)
    assert rep["row_groups_read"] == rep["of_total_row_groups"] == 4
    assert rep["originals_written"] == 7


def test_a_row_without_instruction_or_image_bytes_is_skipped(tmp_path: Path):
    s = tmp_path / "tmp_data/downloads/train-00000-of-00571.parquet"
    _shard(s, [
        ("ok_1", ["Add a hat", "Add a red hat"], _png(), "addition"),
        ("no_instruction", [], _png(), "addition"),
        ("no_bytes", ["Add a hat"], b"", "addition"),
    ])
    rep = FP.fetch(label="omniedit_train", local_parquet=str(s), row_groups=0, root=tmp_path)
    assert rep["originals_written"] == 1


def test_the_task_column_rides_through_for_the_collision_screen(tmp_path: Path):
    """`exclude_tasks` screens `metadata.raw_task`; if it is absent the screen is a silent no-op."""
    s = tmp_path / "tmp_data/downloads/train-00513-of-00571.parquet"
    _shard(s, _rows(2, prefix="task_env_weather", task="env"))
    FP.fetch(label="omniedit_train", local_parquet=str(s), row_groups=0, root=tmp_path)
    recs = io.read_jsonl(
        tmp_path / "data/manifests/samples_omniedit_train_probe.jsonl", SampleRecord)
    assert {r.metadata.raw_task for r in recs} == {"env"}


def test_shard_lockfile_matches_what_is_ingested():
    """The frozen candidate set is a red line, so the lockfile is checked, not trusted.

    Seeded-shuffle-then-take-prefix only yields a superset while the candidate set is fixed;
    the lockfile IS that set. A shard on disk but absent from it (or vice versa) means the
    prefix property no longer holds and any already-paid judge call is at risk.
    """
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "data/provenance/omniedit_shard_lock.json"
    manifest = root / "data/manifests/samples_omniedit_train_probe.jsonl"
    if not (lock_path.exists() and manifest.exists()):
        pytest.skip("built artefacts are git-ignored; nothing to check in a clean checkout")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = {s["file"] for s in lock["shards"]}
    ingested = {Path(r.metadata.source_file).name
                for r in io.iter_jsonl(manifest, SampleRecord)}
    assert ingested == locked
    assert sum(s["rows"] for s in lock["shards"]) == lock["rows_total"]
