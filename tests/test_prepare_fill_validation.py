"""Tests for the FILL v2 P4 manifest: a paired design on newly injected images only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prepare_fill_validation",
                                               REPO / "scripts" / "prepare_fill_validation.py")
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)  # type: ignore[union-attr]


def _records(n_new: int, n_published: int) -> list:
    rows = []
    for i in range(n_new + n_published):
        base = f"b{i:03d}"
        folder = "biased_pair_fill" if i < n_new else "biased"
        for cond in p4.CONDITIONS:
            rows.append({"biased_id": f"{base}__{cond}", "base_sample_id": base,
                         "bias_type": cond,
                         "biased_image_path": f"data/images/{folder}/{cond}/{base}__{cond}.png"})
    return rows


def test_every_chosen_base_carries_all_eight_conditions():
    """The sham floor is only a floor for a cue if it was measured on the same pictures."""
    rows = p4.select_p4(_records(30, 10), n_bases=12)
    bases = {r["base_sample_id"] for r in rows}
    assert len(bases) == 12 and len(rows) == 12 * len(p4.CONDITIONS)
    for base in bases:
        assert {r["bias_type"] for r in rows if r["base_sample_id"] == base} == set(p4.CONDITIONS)


def test_only_newly_injected_images_are_chosen():
    rows = p4.select_p4(_records(12, 40), n_bases=12)
    assert all("biased_pair_fill" in r["biased_image_path"] for r in rows)


def test_the_draw_is_fixed_by_the_seed():
    records = _records(40, 0)
    first = {r["base_sample_id"] for r in p4.select_p4(records, n_bases=10, seed=42)}
    again = {r["base_sample_id"] for r in p4.select_p4(records, n_bases=10, seed=42)}
    other = {r["base_sample_id"] for r in p4.select_p4(records, n_bases=10, seed=7)}
    assert first == again and first != other


def test_a_base_missing_a_condition_is_not_eligible():
    records = [r for r in _records(3, 0) if not (r["base_sample_id"] == "b000" and r["bias_type"] == "sham")]
    rows = p4.select_p4(records, n_bases=2)
    assert "b000" not in {r["base_sample_id"] for r in rows}
    with pytest.raises(ValueError, match="only 2 bases"):
        p4.select_p4(records, n_bases=3)
