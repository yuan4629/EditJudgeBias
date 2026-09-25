"""Shared helpers for manifest building and validation.

Paths in manifests are stored **relative to a project root, POSIX-style** so the
artifacts are portable. A `root` (default: current working directory) resolves
them back to real files for existence checks and image loading.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def default_root() -> Path:
    """Project root that manifest-relative paths resolve against (the CWD)."""
    return Path.cwd()


def to_rel_posix(path: PathLike, root: PathLike) -> str:
    """Render `path` relative to `root` as a POSIX string when possible.

    The lexical relation is tried first, without resolving symlinks, so a corpus
    directory that is a symlink or junction to another disk (e.g. `tmp_data/`) still
    yields `tmp_data/...`. This matters beyond portability: manifest paths seed some of
    the benchmark's sampling, so an absolute path would silently change the draw. Then
    the resolved relation is tried; if `path` is outside `root` either way (e.g. on a
    different drive on Windows), the absolute POSIX path is returned, so the result is
    always a usable string.
    """
    lexical = Path(os.path.normpath(Path(path).absolute()))
    lexical_root = Path(os.path.normpath(Path(root).absolute()))
    try:
        return lexical.relative_to(lexical_root).as_posix()
    except ValueError:
        pass
    p = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def resolve(path: PathLike, root: PathLike) -> Path:
    """Resolve a (possibly manifest-relative) path against `root`."""
    p = Path(path)
    return p if p.is_absolute() else (Path(root) / p)


def find_by_stem(directory: Path, stem: str) -> Optional[Path]:
    """Return the file in `directory` whose stem == `stem`, regardless of suffix.

    I2EBench edited outputs reuse the source stem but may change the extension
    (e.g. input `0001.png` -> qwen output `0001.jpg`), so matching is by stem.
    Returns None if the directory is missing or no file matches.
    """
    if not directory.is_dir():
        return None
    exact = directory / stem
    if exact.is_file():
        return exact
    matches = sorted(p for p in directory.glob(f"{stem}.*") if p.is_file())
    return matches[0] if matches else None


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str) -> str:
    """Filesystem/id-safe slug (keeps alnum, dot, underscore, hyphen)."""
    return _SLUG_RE.sub("-", text).strip("-")
