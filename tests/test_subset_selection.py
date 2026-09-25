"""Judge-subset selection: per-stratum k overrides and the human-anchor block.

Both mechanisms exist to buy something the plain stratified draw cannot, and both
are only safe because they preserve the prefix/superset invariant — an
already-injected or already-paid sample must never fall out of the draw. Every
test here is ultimately about one of those two things.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from edit_judge_bias.data.build_full_manifest import combine, k_resolver, load_config
from edit_judge_bias.data.io import read_jsonl, write_jsonl
from edit_judge_bias.data.schema import SampleRecord


def _img(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "img" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def _flat(tmp_path: Path, sid: str, *, source: str, edit_type: str) -> SampleRecord:
    """A sample on its own original — i.e. a turn of one, deriving no pairs."""
    return SampleRecord(
        sample_id=sid, source_dataset=source, edit_type=edit_type,
        content_category="object",
        original_image_path=_img(tmp_path, f"{sid}_o.png").relative_to(tmp_path),
        edited_image_path=_img(tmp_path, f"{sid}_e.png").relative_to(tmp_path),
        instruction=f"unique instruction {sid}", edit_model=f"m{sid[-1]}",
    )


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _subset(tmp_path: Path) -> list[SampleRecord]:
    return read_jsonl(tmp_path / "sub_s.jsonl", SampleRecord)


def _ids(tmp_path: Path) -> set[str]:
    return {s.sample_id for s in _subset(tmp_path)}


# --------------------------------------------------------------------------- #
# Per-stratum k override                                                       #
# --------------------------------------------------------------------------- #
def test_k_resolver_prefers_the_source_specific_key():
    k_of = k_resolver(20, {"low-level": 60, "I2EBench/low-level": 99})
    assert k_of(("I2EBench", "low-level")) == 99     # most specific wins
    assert k_of(("EBench-18K", "low-level")) == 60   # edit_type applies everywhere
    assert k_of(("I2EBench", "add")) == 20           # base
    assert k_of("not-a-stratum") == 20


def _flat_fixture(tmp_path: Path, *, k=2, overrides=None) -> Path:
    samples = [_flat(tmp_path, f"{source[-1]}{et[0]}{i}", source=source, edit_type=et)
               for source in ("SrcA", "SrcB") for et in ("add", "remove") for i in range(4)]
    write_jsonl(tmp_path / "s1.jsonl", [s for s in samples if s.source_dataset == "SrcA"])
    write_jsonl(tmp_path / "s2.jsonl", [s for s in samples if s.source_dataset == "SrcB"])
    sub = {"seed": 42, "per_source_per_edit_type": k,
           "pairs_per_source_per_edit_type": 2,
           "samples": "sub_s.jsonl", "pairs": "sub_p.jsonl"}
    if overrides is not None:
        sub["per_edit_type_overrides"] = overrides
    return _write_cfg(tmp_path, {
        "sources": [{"name": "SrcA", "samples": "s1.jsonl"},
                    {"name": "SrcB", "samples": "s2.jsonl"}],
        "output": {"samples": "out_s.jsonl", "pairs": "out_p.jsonl"},
        "judge_subset": sub,
    })


def test_deepening_one_edit_type_leaves_every_other_stratum_untouched(tmp_path):
    """The whole point: buy depth in one class without re-drawing — and possibly
    losing — an already-judged sample somewhere else."""
    combine(_flat_fixture(tmp_path, k=2), root=tmp_path)
    before = _ids(tmp_path)
    combine(_flat_fixture(tmp_path, k=2, overrides={"remove": 3}), root=tmp_path)
    after = _ids(tmp_path)

    assert before <= after                                # nothing lost
    assert len(after - before) == 2                       # one per source
    assert all(sid[1] == "r" for sid in after - before)   # and only in `remove`


def test_a_source_specific_override_touches_exactly_one_stratum(tmp_path):
    combine(_flat_fixture(tmp_path, k=2), root=tmp_path)
    before = _ids(tmp_path)
    combine(_flat_fixture(tmp_path, k=2, overrides={"SrcA/remove": 4}), root=tmp_path)
    added = _ids(tmp_path) - before
    assert len(added) == 2 and all(sid.startswith("Ar") for sid in added)


def test_an_overridden_stratum_reports_its_own_shortfall(tmp_path):
    """A silent truncation here would look like a fully-populated cell in §9.4."""
    stats = combine(_flat_fixture(tmp_path, k=2, overrides={"remove": 60}), root=tmp_path)
    shorts = [s for s in stats.subset_strata_short if "remove" in s]
    assert len(shorts) == 2 and all(s.endswith("/60") for s in shorts)


# --------------------------------------------------------------------------- #
# Human-anchor block (group-wise draw)                                         #
# --------------------------------------------------------------------------- #
def _turn_fixture(tmp_path: Path, *, turns=1, models_per_turn=None,
                  require_human_score=True, k=1) -> Path:
    """3 turns per edit_type, 4 scored editors each — EBench's shape in miniature.

    Plus a second source of turn-of-one samples, to prove the block draws only
    from the source it is configured with.
    """
    samples = []
    for et in ("add", "remove"):
        for t in range(3):
            orig = _img(tmp_path, f"turn_{et}_{t}_o.png")
            for m in range(4):
                sid = f"{et[0]}{t}m{m}"
                samples.append(SampleRecord(
                    sample_id=sid, source_dataset="Anchor", edit_type=et,
                    content_category="object",
                    original_image_path=orig.relative_to(tmp_path),
                    edited_image_path=_img(tmp_path, f"{sid}_e.png").relative_to(tmp_path),
                    instruction=f"do {et} {t}", edit_model=f"model{m}",
                    human_score=0.1 * m,
                ))
    others = [_flat(tmp_path, f"o{i}", source="Other", edit_type="add") for i in range(4)]
    write_jsonl(tmp_path / "anchor.jsonl", samples)
    write_jsonl(tmp_path / "other.jsonl", others)
    return _write_cfg(tmp_path, {
        "sources": [{"name": "Anchor", "samples": "anchor.jsonl"},
                    {"name": "Other", "samples": "other.jsonl"}],
        "output": {"samples": "out_s.jsonl", "pairs": "out_p.jsonl"},
        "judge_subset": {
            "seed": 42, "per_source_per_edit_type": k,
            "pairs_per_source_per_edit_type": 2,
            "human_anchor_block": {
                "source": "Anchor", "turns_per_edit_type": turns,
                "models_per_turn": models_per_turn,
                "require_human_score": require_human_score,
            },
            "samples": "sub_s.jsonl", "pairs": "sub_p.jsonl",
        },
    })


def _anchored(tmp_path: Path) -> list[SampleRecord]:
    return [s for s in _subset(tmp_path)
            if s.metadata.model_extra.get("subset_block") == "anchor"]


def test_the_block_takes_whole_turns_so_pairs_can_be_derived(tmp_path):
    stats = combine(_turn_fixture(tmp_path, turns=2), root=tmp_path)
    assert stats.hblock_turns == 4                  # 2 edit_types x 2 turns
    assert stats.hblock_samples == 16               # whole turns: x 4 editors
    assert stats.hblock_pairs_derivable == 4 * 6    # C(4,2) per turn


def test_the_block_only_draws_from_its_configured_source(tmp_path):
    combine(_turn_fixture(tmp_path, turns=3), root=tmp_path)
    assert {s.source_dataset for s in _anchored(tmp_path)} == {"Anchor"}


def test_every_subset_row_is_labelled_breadth_or_anchor(tmp_path):
    """Claim A must be able to exclude the block — it is one source, one shape."""
    combine(_turn_fixture(tmp_path, turns=1), root=tmp_path)
    assert {s.metadata.model_extra.get("subset_block") for s in _subset(tmp_path)} \
        == {"breadth", "anchor"}


def test_raising_turns_per_edit_type_is_a_superset(tmp_path):
    combine(_turn_fixture(tmp_path, turns=1), root=tmp_path)
    before = _ids(tmp_path)
    combine(_turn_fixture(tmp_path, turns=3), root=tmp_path)
    after = _ids(tmp_path)
    assert before <= after and len(after) > len(before)


def test_raising_models_per_turn_is_a_superset(tmp_path):
    combine(_turn_fixture(tmp_path, turns=2, models_per_turn=2), root=tmp_path)
    before = _ids(tmp_path)
    combine(_turn_fixture(tmp_path, turns=2, models_per_turn=4), root=tmp_path)
    assert before <= _ids(tmp_path)


def test_capping_models_per_turn_does_not_pin_the_same_editors_everywhere(tmp_path):
    """`sorted(...)[:n]` would keep model00..model0n in every turn and leave the
    rest of the roster completely unmeasured."""
    combine(_turn_fixture(tmp_path, turns=3, models_per_turn=2), root=tmp_path)
    assert len({s.edit_model for s in _anchored(tmp_path)}) > 2


def test_turns_without_a_human_score_are_skipped(tmp_path):
    cfg_path = _turn_fixture(tmp_path, turns=3)
    samples = read_jsonl(tmp_path / "anchor.jsonl", SampleRecord)
    for s in samples:
        if s.sample_id.startswith("a0"):
            s.human_score = None
    write_jsonl(tmp_path / "anchor.jsonl", samples)
    stats = combine(cfg_path, root=tmp_path)
    assert stats.hblock_turns == 5  # the `add` turn 0 is no longer eligible


def test_the_block_only_ever_adds_to_the_stratified_draw(tmp_path):
    combine(_turn_fixture(tmp_path, turns=0, k=2), root=tmp_path)
    breadth_only = _ids(tmp_path)
    combine(_turn_fixture(tmp_path, turns=3, k=2), root=tmp_path)
    assert breadth_only <= _ids(tmp_path)
