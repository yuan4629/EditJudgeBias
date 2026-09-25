"""Tests for the WP-A6c second-annotator subset and its Cohen's kappa.

Why these and not more: the three properties kappa depends on all fail SILENTLY.

* If the subset were RE-RENDERED instead of copied, the two annotators would look at two
  pictures the moment the compositor changed, and kappa would become a measure of that
  drift.  Nothing in the output would say so.
* If the cue mix drifted from the primary package's, kappa would no longer be an
  agreement rate for the sheet it is quoted next to -- and the placebo share is the part
  that matters most, because `sham` is where a lenient and a strict annotator diverge.
* If the indices were shared, the two sheets would be comparable before either was done,
  and "independent" annotation would be an assertion rather than a property.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "human_validation_package", REPO_ROOT / "scripts" / "human_validation_package.py"
)
pkg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pkg)

LABEL_COL = pkg.LABEL_COL


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def primary(tmp_path: Path) -> Path:
    """A stand-in primary package: 6 cues x 6 items, distinct pixels per item."""
    out = tmp_path / "primary"
    (out / "images").mkdir(parents=True)
    labels, key = [], []
    idx = 0
    for cue in pkg.CUES:
        for j in range(6):
            idx += 1
            fname = f"{idx:03d}.png"
            Image.new("RGB", (8, 8), (idx, idx, idx)).save(out / "images" / fname)
            # A real edit instruction, i.e. one that never names the cue -- the sheet is
            # only blinded if the instruction column cannot give the cue away either.
            labels.append({"index": idx, "image_file": f"images/{fname}",
                           "instruction": f"Remove the object {idx}", LABEL_COL: ""})
            key.append({"index": idx, "image_file": f"images/{fname}", "bias_type": cue,
                        "biased_id": f"b{idx}__{cue}", "base_sample_id": f"s{idx}"})
    _write_csv(out / pkg.LABELS, labels)
    _write_csv(out / pkg.KEY, key)
    return out


def _build(primary: Path, tmp_path: Path, per_cue=2, seed=777) -> Path:
    dst = tmp_path / "iaa"
    args = pkg.argparse.Namespace(
        out_dir=str(primary), iaa_dir=str(dst), per_cue=per_cue, seed=seed
    )
    assert pkg.cmd_build_iaa(args) == 0
    return dst


def test_images_are_byte_identical_copies_not_re_renders(primary: Path, tmp_path: Path):
    dst = _build(primary, tmp_path)
    key = {r["primary_index"]: r for r in _read_csv(dst / pkg.KEY)}
    for pidx, row in key.items():
        src = (primary / _read_csv(primary / pkg.KEY)[int(pidx) - 1]["image_file"]).read_bytes()
        assert (dst / row["image_file"]).read_bytes() == src


def test_cue_mix_matches_the_primary_package(primary: Path, tmp_path: Path):
    dst = _build(primary, tmp_path)
    counts = {}
    for r in _read_csv(dst / pkg.KEY):
        counts[r["bias_type"]] = counts.get(r["bias_type"], 0) + 1
    assert counts == {cue: 2 for cue in pkg.CUES}


def test_rows_are_a_subset_of_the_primary_package_with_no_duplicates(
    primary: Path, tmp_path: Path
):
    dst = _build(primary, tmp_path)
    sub = [r["biased_id"] for r in _read_csv(dst / pkg.KEY)]
    prim = {r["biased_id"] for r in _read_csv(primary / pkg.KEY)}
    assert set(sub) <= prim
    assert len(sub) == len(set(sub))


def test_indices_are_renumbered_not_inherited(primary: Path, tmp_path: Path):
    """Shared indices would let one annotator's remark land on the other's row."""
    dst = _build(primary, tmp_path)
    rows = _read_csv(dst / pkg.KEY)
    assert [r["index"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    assert sum(r["index"] == r["primary_index"] for r in rows) < len(rows)


def test_annotator_sheet_carries_no_cue(primary: Path, tmp_path: Path):
    dst = _build(primary, tmp_path)
    text = (dst / pkg.LABELS).read_text(encoding="utf-8-sig")
    for cue in pkg.CUES:
        assert cue not in text
    assert all(not (r[LABEL_COL] or "") for r in _read_csv(dst / pkg.LABELS))


def test_refuses_when_a_cue_cannot_fill_the_quota(primary: Path, tmp_path: Path):
    dst = tmp_path / "iaa2"
    args = pkg.argparse.Namespace(
        out_dir=str(primary), iaa_dir=str(dst), per_cue=99, seed=777
    )
    assert pkg.cmd_build_iaa(args) == 1


def _label_both(primary: Path, dst: Path, a1: dict, a2: dict):
    """`a1`/`a2` map biased_id -> label; writes the two sheets the kappa step reads."""
    key = _read_csv(dst / pkg.KEY)
    _write_csv(primary / pkg.RESOLVED, [
        {"index": r["primary_index"], "image_file": r["image_file"],
         "bias_type": r["bias_type"], "biased_id": r["biased_id"],
         "base_sample_id": r["base_sample_id"], LABEL_COL: a1[r["biased_id"]]}
        for r in key if r["biased_id"] in a1
    ])
    _write_csv(dst / pkg.LABELS, [
        {"index": r["index"], "image_file": r["image_file"], "instruction": "x",
         LABEL_COL: a2.get(r["biased_id"], "")} for r in key
    ])


def test_kappa_is_one_when_the_two_annotators_agree_everywhere(
    primary: Path, tmp_path: Path, capsys
):
    dst = _build(primary, tmp_path)
    ids = [r["biased_id"] for r in _read_csv(dst / pkg.KEY)]
    labels = {b: ("Yes" if i % 2 else "No") for i, b in enumerate(ids)}
    _label_both(primary, dst, labels, labels)
    args = pkg.argparse.Namespace(out_dir=str(primary), iaa_dir=str(dst))
    assert pkg.cmd_kappa(args) == 0
    out = pkg.json.loads((dst / "iaa_kappa.json").read_text(encoding="utf-8"))
    assert out["cohens_kappa_3level"] == 1.0
    assert out["n_shared"] == len(ids)


def test_kappa_reports_the_binary_collapse_the_gate_actually_uses(
    primary: Path, tmp_path: Path
):
    """No-vs-Slightly disagreement must not read as gate disagreement.

    The paper's preservation gate counts No and Slightly alike; a three-level kappa that
    is dragged down purely by that boundary would misdescribe the gate's reliability.
    """
    dst = _build(primary, tmp_path)
    ids = [r["biased_id"] for r in _read_csv(dst / pkg.KEY)]
    a1 = {b: ("No" if i % 2 else "Yes") for i, b in enumerate(ids)}
    a2 = {b: ("Slightly" if i % 2 else "Yes") for i, b in enumerate(ids)}
    _label_both(primary, dst, a1, a2)
    args = pkg.argparse.Namespace(out_dir=str(primary), iaa_dir=str(dst))
    assert pkg.cmd_kappa(args) == 0
    out = pkg.json.loads((dst / "iaa_kappa.json").read_text(encoding="utf-8"))
    assert out["cohens_kappa_3level"] < out["cohens_kappa_binary_yes_vs_rest"]
    assert out["cohens_kappa_binary_yes_vs_rest"] == 1.0


def test_kappa_uses_only_rows_both_annotators_finished(primary: Path, tmp_path: Path):
    dst = _build(primary, tmp_path)
    ids = [r["biased_id"] for r in _read_csv(dst / pkg.KEY)]
    a1 = {b: "No" for b in ids}
    a2 = {b: "No" for b in ids[:4]}  # annotator 2 stopped early
    _label_both(primary, dst, a1, a2)
    args = pkg.argparse.Namespace(out_dir=str(primary), iaa_dir=str(dst))
    assert pkg.cmd_kappa(args) == 0
    out = pkg.json.loads((dst / "iaa_kappa.json").read_text(encoding="utf-8"))
    assert out["n_shared"] == 4


# --------------------------------------------------------------------------- #
# WP-F5 — two tracked files, one quantity, two vintages                        #
# --------------------------------------------------------------------------- #
def test_the_package_snapshot_agrees_with_the_metrics_table():
    """`data/human_validation_v2_iaa/iaa_kappa.json` is written by
    `human_validation_package.py kappa`; `results/v2/metrics/human_validation_iaa.json` is
    written by `scripts/06_build_tables.sh`.  Nothing connected them, and on 2026-08-19 they were found
    two ROUNDS apart -- the package file still said kappa = 0.4074 (round 1) while the
    metrics table said 0.8930 (round 3), both tracked, both looking authoritative.

    Same shape as the `labels.csv` / `unblind` trap that produced a mixed-vintage kappa
    twice in one day: whenever one artefact is regenerated by the main pipeline and
    another by a side command, the side one silently becomes a fossil.  This test is the
    connection, and the fix when it fires is to re-run:

        python scripts/human_validation_package.py kappa
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pkg = root / "data" / "human_validation_v2_iaa" / "iaa_kappa.json"
    tbl = root / "results" / "v2" / "metrics" / "human_validation_iaa.json"
    if not (pkg.exists() and tbl.exists()):
        pytest.skip("IAA artefacts not built")

    a = json.loads(pkg.read_text(encoding="utf-8"))
    b = json.loads(tbl.read_text(encoding="utf-8"))
    assert a["n_shared"] == b["n_shared"], "the two files describe different label sets"
    assert round(a["cohens_kappa_3level"], 4) == round(b["three_level"]["cohens_kappa"], 4), (
        "iaa_kappa.json is stale -- run `python scripts/human_validation_package.py kappa`"
    )
    assert round(a["cohens_kappa_binary_yes_vs_rest"], 4) == round(
        b["binary"]["yes_vs_rest"]["cohens_kappa"], 4)
