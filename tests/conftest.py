"""Shared test fixtures/helpers for the data layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def touch_image(path: Path) -> Path:
    """Create a tiny placeholder image file (content irrelevant to M1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic; enough for existence checks
    return path


def make_rgb_image(path: Path, size=(120, 90), color=(128, 100, 80)) -> Path:
    """Write a real decodable RGB image (for M2 bias-injector tests)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def make_mini_editbench(
    root: Path,
    categories: Dict[str, List[dict]],
    models: Dict[str, List[str]] | None = None,
    *,
    edited_ext: str = ".jpg",
) -> Path:
    """Build a miniature EditBench tree under `root` and return its dataset root.

    `categories`: {category_name: [entry_dict, ...]} where each entry has at least
    "image" and "ori_exp" (and optionally "type"). `models`: {model: [image_names
    to SKIP]} to simulate missing edited outputs (default: all present, 2 models).
    Original inputs keep the json extension; edited outputs use `edited_ext` to
    exercise stem-matching.
    """
    models = models if models is not None else {"modelX": [], "modelY": []}
    eb = root / "EditBench"
    for category, entries in categories.items():
        cat_dir = eb / "EditData" / category
        (cat_dir / "input").mkdir(parents=True, exist_ok=True)
        data = {str(i + 1): e for i, e in enumerate(entries)}
        (cat_dir / f"{category}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        for e in entries:
            touch_image(cat_dir / "input" / e["image"])
            stem = Path(e["image"]).stem
            for model, skip in models.items():
                if e["image"] in skip:
                    continue
                touch_image(eb / "EditResult" / category / model / f"{stem}{edited_ext}")
    return eb


def mini_config(editbench_root: str = "EditBench", **overrides) -> dict:
    cfg = {
        "source_dataset": "I2EBench",
        "editbench_root": editbench_root,
        "instruction_field": "ori_exp",
        "edit_type_map": {"ObjectRemoval": "remove", "ColorAlteration": "color"},
        "content_category_map": {
            "human": "human",
            "animal": "animal",
            "object": "object",
            "scenery": "scenery",
            "plant": "object",
            "global": "global",
        },
        "default_content_category": "global",
        "edit_models": [],
        "pilot": {"seed": 42, "groups_per_edit_type": 2, "max_models_per_group": 10},
        "output": {
            "samples_full": "data/manifests/samples.jsonl",
            "samples_pilot": "data/manifests/samples_pilot.jsonl",
        },
    }
    cfg.update(overrides)
    return cfg
