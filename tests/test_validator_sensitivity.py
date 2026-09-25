"""Tests for the WP-A4b validator-sensitivity table.

The load-bearing one is `test_a_rubber_stamp_scores_perfectly_on_sham_and_is_caught_here`:
a validator that never flags anything has a PERFECT specificity floor, so `sham` alone
certifies it.  This table exists precisely to catch that, and the test asserts both halves
of the contrast on one instrument.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edit_judge_bias.experiments.build_validator_sensitivity import (
    CONDITIONS,
    REFERENCE_CONDITION,
    WEAK_DAMAGE,
    build_validator_sensitivity,
    write_validator_sensitivity,
)


def _write_manifest(path: Path, per_condition: dict[str, dict[str, float]]) -> None:
    lines = []
    for condition, changes in per_condition.items():
        for base, change in changes.items():
            lines.append(
                json.dumps(
                    {
                        "biased_id": f"{base}__{condition}",
                        "base_sample_id": base,
                        "bias_type": condition,
                        "bias_params": {
                            "mode": "blur" if condition.endswith("blur") else "revert",
                            "severity": 1.0,
                            "mean_abs_change": change,
                        },
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_verdicts(path: Path, verdicts: dict[str, bool], *, parse_ok: set[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for biased_id, ok in verdicts.items():
        row = {"biased_id": biased_id, "pass": ok, "parse_success": True}
        if parse_ok is not None and biased_id not in parse_ok:
            row = {"biased_id": biased_id, "pass": None, "parse_success": False}
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup(tmp_path: Path, *, damage: dict[str, bool], sham: dict[str, bool],
           changes: dict[str, float] | None = None) -> tuple[Path, Path, Path]:
    """One condition (severity 1.0) unless `changes` says otherwise."""
    changes = changes or {b.rpartition("__")[0]: 10.0 for b in damage}
    manifest = tmp_path / "control.jsonl"
    _write_manifest(manifest, {"edit_damage_100": changes})
    control_dir = tmp_path / "control_results"
    published_dir = tmp_path / "published"
    _write_verdicts(control_dir / "validation__v1.jsonl", damage)
    _write_verdicts(published_dir / "validation__v1.jsonl", sham)
    return manifest, control_dir, published_dir


def test_perfect_instrument(tmp_path: Path):
    bases = [f"s{i}" for i in range(10)]
    damage = {f"{b}__edit_damage_100": False for b in bases}   # flagged every damaged image
    sham = {f"{b}__sham": True for b in bases}                 # passed every null image
    args = _setup(tmp_path, damage=damage, sham=sham)
    (row,) = build_validator_sensitivity(*args)

    assert row["n_evaluable_fixed"] == 10
    assert row["sensitivity_fixed"] == 1.0
    assert row["false_flag_rate_same_images"] == 0.0
    assert row["discrimination"] == 1.0
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (10, 0)
    assert row["discriminates"] is True


def test_a_rubber_stamp_scores_perfectly_on_sham_and_is_caught_here(tmp_path: Path):
    """★ THE POINT OF THE ARM.

    A validator that answers "preserved" to everything has a false-flag floor of 0.000 --
    better than any real instrument -- so the `sham` arm alone would certify it as the
    best-calibrated validator in the panel.  Sensitivity is what exposes it.
    """
    bases = [f"s{i}" for i in range(10)]
    damage = {f"{b}__edit_damage_100": True for b in bases}    # passed the DESTROYED edits
    sham = {f"{b}__sham": True for b in bases}
    args = _setup(tmp_path, damage=damage, sham=sham)
    (row,) = build_validator_sensitivity(*args)

    # The specificity half looks flawless...
    assert row["false_flag_rate_same_images"] == 0.0
    # ...and the sensitivity half is zero.
    assert row["sensitivity_fixed"] == 0.0
    assert row["discrimination"] == 0.0
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (0, 0)
    assert row["discriminates"] is not True


def test_weak_damage_images_leave_the_denominator_and_are_reported(tmp_path: Path):
    """17 of the real 110 change by < 1/255 even at full revert.  On those the validator
    is RIGHT to say 'preserved', so counting them as misses would manufacture the very
    insensitivity being measured."""
    strong = {f"s{i}": 10.0 for i in range(6)}
    weak = {f"w{i}": WEAK_DAMAGE / 2 for i in range(4)}
    changes = {**strong, **weak}
    damage = {f"{b}__edit_damage_100": (b.startswith("w")) for b in changes}  # flags only strong
    sham = {f"{b}__sham": True for b in changes}
    args = _setup(tmp_path, damage=damage, sham=sham, changes=changes)
    (row,) = build_validator_sensitivity(*args)

    assert row["n_judged"] == 10
    assert row["n_weak_excluded"] == 4
    assert row["n_evaluable_fixed"] == 6
    assert row["sensitivity_fixed"] == 1.0          # 6/6, not 6/10
    assert row["n_flagged_among_weak"] == 0


def test_fixed_and_own_denominators_diverge_on_a_weak_rung(tmp_path: Path):
    """The threshold curve needs one denominator to stay comparable rung to rung; the
    validator deserves another that does not charge it for invisible damage.  Both ship."""
    bases = [f"s{i}" for i in range(8)]
    manifest = tmp_path / "control.jsonl"
    _write_manifest(
        manifest,
        {
            # every image is evaluable at full strength ...
            "edit_damage_100": {b: 8.0 for b in bases},
            # ... but at quarter strength half of them fall below the threshold
            "edit_damage_25": {b: (2.0 if i < 4 else 0.4) for i, b in enumerate(bases)},
        },
    )
    control_dir = tmp_path / "control_results"
    published_dir = tmp_path / "published"
    _write_verdicts(
        control_dir / "validation__v1.jsonl",
        {**{f"{b}__edit_damage_100": False for b in bases},
         # at 25%, flags the four visible ones and passes the four invisible ones
         **{f"{b}__edit_damage_25": (i >= 4) for i, b in enumerate(bases)}},
    )
    _write_verdicts(published_dir / "validation__v1.jsonl", {f"{b}__sham": True for b in bases})

    rows = {r["condition"]: r for r in build_validator_sensitivity(manifest, control_dir, published_dir)}
    weak = rows["edit_damage_25"]
    assert weak["n_evaluable_fixed"] == 8 and weak["sensitivity_fixed"] == 0.5
    assert weak["n_evaluable_own"] == 4 and weak["sensitivity_own"] == 1.0
    # The headline rung is identical under both, which is why it is the headline.
    strong = rows["edit_damage_100"]
    assert strong["sensitivity_fixed"] == strong["sensitivity_own"] == 1.0


def test_anti_discrimination_is_not_reported_as_discrimination(tmp_path: Path):
    """A validator that flags the SHAM more often than the destroyed edit is worse than
    useless, and a bare q < 0.05 cannot tell that apart from the good direction."""
    bases = [f"s{i}" for i in range(12)]
    damage = {f"{b}__edit_damage_100": True for b in bases}          # passed all damage
    sham = {f"{b}__sham": (i >= 10) for i, b in enumerate(bases)}    # flagged 10 null images
    args = _setup(tmp_path, damage=damage, sham=sham)
    (row,) = build_validator_sensitivity(*args)

    assert row["mcnemar_c"] == 10 and row["mcnemar_b"] == 0
    assert row["mcnemar_q"] is not None and row["mcnemar_q"] < 0.05
    assert row["discriminates"] is False, "significance alone must not read as discrimination"


def test_parse_failures_are_dropped_not_counted_as_a_verdict(tmp_path: Path):
    bases = [f"s{i}" for i in range(10)]
    damage = {f"{b}__edit_damage_100": False for b in bases}
    ok = {f"{b}__edit_damage_100" for b in bases[:7]}
    changes = {b: 10.0 for b in bases}
    manifest = tmp_path / "control.jsonl"
    _write_manifest(manifest, {"edit_damage_100": changes})
    control_dir = tmp_path / "control_results"
    published_dir = tmp_path / "published"
    _write_verdicts(control_dir / "validation__v1.jsonl", damage, parse_ok=ok)
    _write_verdicts(published_dir / "validation__v1.jsonl", {f"{b}__sham": True for b in bases})

    (row,) = build_validator_sensitivity(manifest, control_dir, published_dir)
    assert row["n_judged"] == 7, "a row that did not parse carries no verdict"
    assert row["sensitivity_fixed"] == 1.0


def test_missing_reference_condition_raises_rather_than_guessing(tmp_path: Path):
    manifest = tmp_path / "control.jsonl"
    _write_manifest(manifest, {"edit_damage_25": {"s0": 2.0}})
    with pytest.raises(ValueError, match="fixed denominator"):
        build_validator_sensitivity(manifest, tmp_path, tmp_path)


def test_absent_sham_file_leaves_the_paired_columns_blank(tmp_path: Path):
    """A missing floor must not silently become a floor of zero -- that would turn every
    sensitivity into a spuriously perfect discrimination."""
    bases = [f"s{i}" for i in range(5)]
    manifest = tmp_path / "control.jsonl"
    _write_manifest(manifest, {"edit_damage_100": {b: 9.0 for b in bases}})
    control_dir = tmp_path / "control_results"
    _write_verdicts(control_dir / "validation__v1.jsonl",
                    {f"{b}__edit_damage_100": False for b in bases})
    (row,) = build_validator_sensitivity(manifest, control_dir, tmp_path / "nonexistent")

    assert row["sensitivity_fixed"] == 1.0
    assert row["false_flag_rate_same_images"] is None
    assert row["discrimination"] is None


def test_csv_roundtrip_has_every_declared_column(tmp_path: Path):
    bases = [f"s{i}" for i in range(4)]
    args = _setup(
        tmp_path,
        damage={f"{b}__edit_damage_100": False for b in bases},
        sham={f"{b}__sham": True for b in bases},
    )
    rows = build_validator_sensitivity(*args)
    out = write_validator_sensitivity(rows, tmp_path / "out.csv")
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "sensitivity_fixed" in header and "discriminates" in header
    assert "n_weak_excluded" in header, "the excluded count must travel with the rate"


def test_conditions_tuple_matches_the_shipped_control():
    """If a rung is added to the ladder on disk and not here, its rows vanish silently."""
    manifest = Path("data/manifests/biased_samples_editdamage_v2.jsonl")
    if not manifest.exists():
        pytest.skip("control manifest not built")
    on_disk = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            on_disk.add(json.loads(line)["bias_type"])
    assert on_disk == set(CONDITIONS)


# --------------------------------------------------------------------------- #
# Frozen-CSV guards.  The write-up makes three QUALITATIVE
# claims off this table.  Prose cannot be tested; the table it was read off can.
# If a rebuild flips any of these, the write-up has to be revisited -- which is
# exactly the failure mode that shipped a falsified sentence inside figure 8's
# hardcoded title before it was caught.
# --------------------------------------------------------------------------- #
FROZEN = Path("results/v2_control/metrics/validator_sensitivity_control.csv")


def _frozen_rows():
    import csv as _csv
    if not FROZEN.exists():  # pragma: no cover - only in a tree without the arm
        pytest.skip(f"{FROZEN} not built")
    with FROZEN.open(encoding="utf-8") as fh:
        return {(r["validator_model"], r["condition"]): r for r in _csv.DictReader(fh)}


def test_frozen_the_two_gating_validators_are_not_the_same_instrument():
    """The headline: one gating validator discriminates, the other does not."""
    rows = _frozen_rows()
    gem = rows[("gemini-3.5-flash", REFERENCE_CONDITION)]
    mini = rows[("gpt-4o-mini", REFERENCE_CONDITION)]
    assert gem["discriminates"] == "True"
    assert mini["discriminates"] == "False", (
        "gpt-4o-mini now discriminates on a full revert -- the published reading "
        "says it does not, and the 'division of labour, not disagreement' "
        "reading rests on it"
    )
    assert float(gem["sensitivity_fixed"]) > 0.9 > float(mini["sensitivity_fixed"])


def test_frozen_the_main_validator_has_a_monotone_dose_response():
    """A curve is harder to fake than a point; §5.7.3 quotes 0.495 -> 0.742 -> 0.946."""
    rows = _frozen_rows()
    ladder = [
        float(rows[("gemini-3.5-flash", c)]["sensitivity_fixed"])
        for c in ("edit_damage_25", "edit_damage_50", "edit_damage_100")
    ]
    assert ladder == sorted(ladder), f"dose-response no longer monotone: {ladder}"


def test_frozen_the_headline_does_not_depend_on_the_weak_damage_exclusion():
    """§17.4: 94.6% vs 3.2% and 88.2% vs 3.6% must tell the same story.

    The exclusion rule was justified as 'these images are indistinguishable', which
    the data then contradicted (gemini flags 9 of the 17).  The rule therefore has
    to be non-load-bearing, and that is a property of the numbers, not an opinion.
    """
    rows = _frozen_rows()
    for name in ("gemini-3.5-flash", "gpt-4o-mini"):
        r = rows[(name, REFERENCE_CONDITION)]
        n_all = int(r["n_evaluable_fixed"]) + int(r["n_weak_excluded"])
        all_rate = (int(r["n_flagged_fixed"]) + int(r["n_flagged_among_weak"])) / n_all
        rows[(name, "__all__")] = all_rate
    assert rows[("gemini-3.5-flash", "__all__")] > 0.5 > rows[("gpt-4o-mini", "__all__")]
