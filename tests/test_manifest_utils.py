"""Tests for the manifest path helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from edit_judge_bias.data.manifest_utils import resolve, to_rel_posix


def test_a_path_inside_the_root_is_stored_relative(tmp_path: Path):
    (tmp_path / "tmp_data").mkdir()
    assert to_rel_posix(tmp_path / "tmp_data" / "a.png", tmp_path) == "tmp_data/a.png"


def test_a_path_outside_the_root_falls_back_to_absolute(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    out = to_rel_posix(tmp_path / "elsewhere" / "a.png", root)
    assert Path(out).is_absolute()


def test_relative_paths_round_trip_through_resolve(tmp_path: Path):
    rel = to_rel_posix(tmp_path / "data" / "x.png", tmp_path)
    assert resolve(rel, tmp_path) == tmp_path / "data" / "x.png"


def test_a_symlinked_corpus_directory_still_yields_relative_paths(tmp_path: Path):
    """Manifest paths seed part of the sampling, so a corpus kept on another disk behind a
    symlink must produce the same `tmp_data/...` strings as a real directory."""
    target = tmp_path / "other_disk" / "corpus"
    target.mkdir(parents=True)
    (target / "a.png").write_bytes(b"x")
    root = tmp_path / "project"
    root.mkdir()
    try:
        os.symlink(target, root / "tmp_data", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this system")
    assert to_rel_posix(root / "tmp_data" / "a.png", root) == "tmp_data/a.png"
