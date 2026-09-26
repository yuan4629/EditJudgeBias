"""Tests for the WP-A4c C-class prompt-ablation table.

The load-bearing one is `test_a_cue_at_the_ceiling_is_uninformative_not_null`: two of the
four cues sit at a published pass rate of 1.000 and cannot move up, so their `b=0` is
arithmetic rather than evidence.  Reading it as a null would be reading a ceiling as a null.
"""

from __future__ import annotations

import json
from pathlib import Path

from edit_judge_bias.experiments.build_prompt_ablation import (
    C_CLASS,
    FIELDS,
    build_prompt_ablation,
    write_prompt_ablation,
)


def _manifest(path: Path, per_cue: dict[str, int]) -> None:
    lines = []
    for cue, n in per_cue.items():
        for i in range(n):
            lines.append(json.dumps({"biased_id": f"img{i}__{cue}", "bias_type": cue}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verdicts(path: Path, verdicts: dict[str, bool], *, unparsed: set[str] = frozenset()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for bid, ok in verdicts.items():
        if bid in unparsed:
            lines.append(json.dumps({"biased_id": bid, "pass": None, "parse_success": False}))
        else:
            lines.append(json.dumps({"biased_id": bid, "pass": ok, "parse_success": True}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_naming_the_overlays_as_cosmetic_moves_the_cue_toward_pass(tmp_path: Path):
    """The outcome that would falsify §5.7: the 0.918 was the template, not the edit."""
    man = tmp_path / "m.jsonl"
    _manifest(man, {"zoom_inset": 20})
    ids = [f"img{i}__zoom_inset" for i in range(20)]
    # published fails 9 of 20; the ablation passes 8 of those 9 and flips nothing back
    _verdicts(tmp_path / "pub.jsonl", {i: (n >= 9) for n, i in enumerate(ids)})
    _verdicts(tmp_path / "abl.jsonl", {i: (n >= 1) for n, i in enumerate(ids)})

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert row["n_paired"] == 20
    assert row["published_pass_rate"] == 0.55 and row["ablation_pass_rate"] == 0.95
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (8, 0)
    assert row["moved_toward_pass"] is True
    assert row["interpretable"] is True
    assert row["mcnemar_q"] is not None and row["mcnemar_q"] < 0.05


def test_no_movement_leaves_the_published_finding_standing(tmp_path: Path):
    man = tmp_path / "m.jsonl"
    _manifest(man, {"zoom_inset": 20})
    ids = [f"img{i}__zoom_inset" for i in range(20)]
    same = {i: (n >= 9) for n, i in enumerate(ids)}
    _verdicts(tmp_path / "pub.jsonl", same)
    _verdicts(tmp_path / "abl.jsonl", same)

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (0, 0)
    assert row["mcnemar_p"] is None and row["mcnemar_q"] is None
    assert row["moved_toward_pass"] is False
    # ...but it IS interpretable, because there was headroom to move into.
    assert row["headroom"] == 9 and row["interpretable"] is True


def test_a_cue_at_the_ceiling_is_uninformative_not_null(tmp_path: Path):
    """★ THE POINT OF THE MODULE.

    `region_annotation` and `detail_caption` are at 1.000 in the published run.  They
    cannot move up, so `b=0` is forced by arithmetic.  The row must say so rather than
    contributing a comfortable-looking null to the ablation's conclusion.
    """
    man = tmp_path / "m.jsonl"
    _manifest(man, {"region_annotation": 20})
    ids = [f"img{i}__region_annotation" for i in range(20)]
    _verdicts(tmp_path / "pub.jsonl", {i: True for i in ids})
    _verdicts(tmp_path / "abl.jsonl", {i: True for i in ids})

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert row["published_pass_rate"] == 1.0
    assert row["headroom"] == 0
    assert row["interpretable"] is False
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (0, 0)


def test_the_ceiling_flag_still_reports_movement_the_wrong_way(tmp_path: Path):
    """A ceiling blocks upward movement only.  If the ablation makes a 1.000 cue WORSE,
    that is real and is the signal that the ablation changed more than intended."""
    man = tmp_path / "m.jsonl"
    _manifest(man, {"detail_caption": 20})
    ids = [f"img{i}__detail_caption" for i in range(20)]
    _verdicts(tmp_path / "pub.jsonl", {i: True for i in ids})
    _verdicts(tmp_path / "abl.jsonl", {i: (n >= 6) for n, i in enumerate(ids)})

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert row["interpretable"] is False          # cannot go UP
    assert (row["mcnemar_b"], row["mcnemar_c"]) == (0, 6)   # but it went DOWN, and we see it
    assert row["delta"] == -0.3


def test_only_images_judged_under_BOTH_prompts_are_paired(tmp_path: Path):
    """An unpaired image would turn the within-image contrast into a between-sample one --
    the exact defect WP-A3 spent $36 fixing one level up."""
    man = tmp_path / "m.jsonl"
    _manifest(man, {"zoom_inset": 10})
    ids = [f"img{i}__zoom_inset" for i in range(10)]
    _verdicts(tmp_path / "pub.jsonl", {i: True for i in ids})
    _verdicts(tmp_path / "abl.jsonl", {i: True for i in ids[:6]})

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert row["n_paired"] == 6


def test_parse_failures_are_not_verdicts(tmp_path: Path):
    man = tmp_path / "m.jsonl"
    _manifest(man, {"zoom_inset": 10})
    ids = [f"img{i}__zoom_inset" for i in range(10)]
    _verdicts(tmp_path / "pub.jsonl", {i: True for i in ids})
    _verdicts(tmp_path / "abl.jsonl", {i: True for i in ids}, unparsed={ids[0], ids[1]})

    (row,) = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    assert row["n_paired"] == 8


def test_csv_roundtrip_has_every_declared_column(tmp_path: Path):
    import csv as _csv
    man = tmp_path / "m.jsonl"
    _manifest(man, {c: 4 for c in C_CLASS})
    ids = [f"img{i}__{c}" for c in C_CLASS for i in range(4)]
    _verdicts(tmp_path / "pub.jsonl", {i: True for i in ids})
    _verdicts(tmp_path / "abl.jsonl", {i: True for i in ids})

    rows = build_prompt_ablation(man, tmp_path / "pub.jsonl", tmp_path / "abl.jsonl")
    out = tmp_path / "out.csv"
    write_prompt_ablation(rows, out)
    with out.open(encoding="utf-8") as fh:
        got = list(_csv.DictReader(fh))
    assert len(got) == len(C_CLASS)
    assert set(got[0]) == set(FIELDS)
