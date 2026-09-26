"""Guards for the five defects found in the pre-spend code review (2026-07-26).

Each test here corresponds to a way the v2 judge grid could have quietly produced
wrong numbers while every existing test stayed green. They are grouped in one file
because they share a single cause: the v2 grid reuses the pilot's identifiers,
manifests and runners, and "reuse" and "collision" look identical until you check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.run_scoring_judge import (
    _pick,
    matches_filter,
    resolve_prompt_style,
)
from edit_judge_bias.metrics.agreement import compute_agreement
from edit_judge_bias.metrics.scoring_metrics import (
    check_single_scale,
    compute_score_shifts,
    is_retest_repeat,
    retest_pairs,
)

REPO = Path(__file__).resolve().parents[1]
V2_ARMS = [
    "scoring_breadth_v2.yaml",
    "scoring_anchor_v2.yaml",
    "scoring_retest_v2.yaml",
    "pairwise_v2.yaml",
    "scoring_smoke_v2.yaml",
]


def _res(rid, sample_id, score, *, scale=10, bias=None, model="j", repeat=None):
    return JudgeResult(
        result_id=rid, judge_model=model, task_type="scoring",
        prompt_type="vanilla_scoring", raw_response_path="r.txt", parse_success=True,
        sample_id=sample_id, overall_score=score, score_scale=scale, bias_type=bias,
        biased_id=f"{sample_id}__{bias}" if bias else None,
        bias_params={"repeat_index": repeat} if repeat else {},
    )


# --------------------------------------------------------------------------- #
# 1. the 1-5 pilot must never pool with the 1-10 grid                          #
# --------------------------------------------------------------------------- #
def test_unstamped_pilot_rows_cannot_pool_with_stamped_grid_rows():
    """The pilot predates `score_scale`, so its rows carry None. Treating None as
    "compatible with anything" is exactly the silent failure: the two eras share
    129 sample_ids, so a mixed manifest would average 1-5 and 1-10 answers."""
    mixed = [_res("a", "s1", 3, scale=None), _res("b", "s2", 7, scale=10)]
    with pytest.raises(ValueError, match="score scales"):
        check_single_scale(mixed, label="scoring results")


def test_an_all_pilot_manifest_still_loads():
    """Re-analysing the archived pilot on its own must stay possible."""
    assert check_single_scale([_res("a", "s1", 3, scale=None)]) is None


# --------------------------------------------------------------------------- #
# 2. the v2 grid writes to its own tree                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", V2_ARMS)
def test_v2_arms_never_write_into_the_pilot_result_tree(name):
    cfg = yaml.safe_load((REPO / "configs/experiment" / name).read_text(encoding="utf-8"))
    assert cfg["results_dir"] == "results/v2", (
        f"{name} would append 1-10 rows to the pilot's manifests, where "
        "load_done_ids would also skip the 129 shared sample_ids as already judged"
    )
    assert cfg["raw_dir"].startswith("results/v2/"), (
        f"{name} would overwrite the pilot's raw .txt responses for shared ids"
    )


def test_validator_manifest_keeps_the_prefix_aggregate_globs_for():
    """`aggregate_results._passed_biased_ids` globs `validation__*.jsonl`; a stem
    like `validation_full_v2__...` does not match it, and the QC table would have
    been built from the pilot's passed ids instead of this run's."""
    cfg = yaml.safe_load(
        (REPO / "configs/experiment/quality_validation_full_v2.yaml").read_text(encoding="utf-8")
    )
    out = Path(cfg["out_manifest"])
    assert out.name.startswith("validation__")
    assert out.parent.as_posix() == "results/v2/quality"


# --------------------------------------------------------------------------- #
# 3. retest repeats are not baselines                                          #
# --------------------------------------------------------------------------- #
def test_a_retest_repeat_never_replaces_the_baseline_it_repeats():
    """Both rows carry bias_type=None and the same sample_id, and both land in the
    unbiased manifest. A last-write-wins lookup would hand every shift computation
    the retest answer for 200 of the breadth block's 611 samples."""
    originals = [
        _res("score::j::vanilla::s1", "s1", 8),
        _res("score::j::vanilla::s1::rep2", "s1", 3, repeat=2),
    ]
    biased = [_res("score::j::vanilla::s1__padding", "s1", 8, bias="padding")]
    (stat,) = compute_score_shifts(originals, biased)
    assert stat.mean_original == 8.0 and stat.mean_shift == 0.0


def test_retest_rows_are_hidden_from_metrics_but_still_readable():
    rows = [
        _res("score::j::vanilla::s1", "s1", 8),
        _res("score::j::vanilla::s1::rep2", "s1", 6, repeat=2),
        _res("score::j::vanilla::s2", "s2", 5),
    ]
    assert [is_retest_repeat(r) for r in rows] == [False, True, False]
    # s2 was asked once, so it is not a retest observation at all.
    assert retest_pairs(rows) == {("j", "s1"): [8, 6]}


def test_the_retest_arm_config_still_forbids_the_cache():
    cfg = yaml.safe_load(
        (REPO / "configs/experiment/scoring_retest_v2.yaml").read_text(encoding="utf-8")
    )
    assert cfg["repeat"] == 2
    assert cfg["judge_overrides"]["cache_dir"] is None
    assert cfg["score_biased"] is False


# --------------------------------------------------------------------------- #
# 4. the VIEScore judge keeps its rubric                                       #
# --------------------------------------------------------------------------- #
def test_the_viescore_judge_overrides_an_arms_vanilla_style():
    """The arm configs pin vanilla and are run across all five judges with
    `--judge-config`. Without this precedence the §2.1 dedicated vision judge
    becomes a second plain gpt-4o, stamped `vanilla_scoring`."""
    judge = yaml.safe_load(
        (REPO / "configs/judge/gpt4o_viescore.yaml").read_text(encoding="utf-8")
    )
    assert judge["scoring_prompt_style"] == "viescore"
    assert resolve_prompt_style({"prompt_style": "vanilla"}, judge, None) == "viescore"


def test_other_judges_keep_the_arms_style_and_the_cli_beats_everything():
    plain = yaml.safe_load((REPO / "configs/judge/gpt5_5.yaml").read_text(encoding="utf-8"))
    assert resolve_prompt_style({"prompt_style": "vanilla"}, plain, None) == "vanilla"
    viescore = {"scoring_prompt_style": "viescore"}
    assert resolve_prompt_style({"prompt_style": "vanilla"}, viescore, "bias_aware") == "bias_aware"


# --------------------------------------------------------------------------- #
# 5. agreement compares the same items before and after                        #
# --------------------------------------------------------------------------- #
def _sample(sid, turn, human, **meta):
    return SampleRecord(
        sample_id=sid, source_dataset="EBench-18K", edit_type="add",
        content_category="object", original_image_path=f"img/{turn}.png",
        instruction=f"instruction {turn}", edit_model=sid, human_score=human,
        edited_image_path=f"edit/{sid}.png", metadata=meta,
    )


# --------------------------------------------------------------------------- #
# 6. subset filtering and the smoke batch's sample pick                        #
# --------------------------------------------------------------------------- #
def test_filter_sees_declared_metadata_fields_not_just_extras():
    """SampleMetadata declares has_mask/mask_path/original_* and allows extras for
    the rest. Reading only `model_extra` made a filter on a declared field match
    nothing, which reads as "that block is empty" rather than as a mistake."""
    masked = _sample("s1", "t", 0.5, has_mask=True, subset_block="breadth")
    plain = _sample("s2", "t", 0.5, has_mask=False, subset_block="breadth")
    assert matches_filter(masked, {"has_mask": True})
    assert not matches_filter(plain, {"has_mask": True})
    # and the extras path still works, including list membership
    assert matches_filter(masked, {"subset_block": ["breadth", "anchor"]})


def test_spread_walks_the_block_while_prefix_stays_at_its_head():
    """The manifest is laid out stratum by stratum, so a prefix of 6 is 6 samples
    from one (source, edit_type) cell — a bad basis for the smoke batch's price
    measurement, which tracks image size."""
    block = [_sample(f"s{i}", f"t{i}", 0.5) for i in range(60)]
    assert [s.sample_id for s in _pick(block, 6, "prefix")] == [f"s{i}" for i in range(6)]
    assert [s.sample_id for s in _pick(block, 6, "spread")] == [
        "s0", "s10", "s20", "s30", "s40", "s50"
    ]


def test_pick_never_runs_off_the_end_or_accepts_a_typo():
    block = [_sample(f"s{i}", "t", 0.5) for i in range(3)]
    assert len(_pick(block, 10, "spread")) == 3
    with pytest.raises(ValueError, match="sample_pick"):
        _pick(block, 2, "sprad")


def test_the_smoke_arm_actually_spans_the_pool():
    """It is the batch that decides which cost tier the grid is in, so it has to
    see more than one resolution and more than one source."""
    cfg = yaml.safe_load(
        (REPO / "configs/experiment/scoring_smoke_v2.yaml").read_text(encoding="utf-8")
    )
    assert cfg["sample_pick"] == "spread"
    assert cfg["results_dir"] == "results/v2"
    manifest = REPO / "data/manifests/samples_judge_v2.jsonl"
    if not manifest.exists():
        pytest.skip("built manifests are git-ignored; run scripts/01_build_manifests.sh")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    breadth = [r for r in rows if (r["metadata"] or {}).get("subset_block") == "breadth"]
    step = len(breadth) / cfg["max_samples"]
    picked = [breadth[int(i * step)] for i in range(cfg["max_samples"])]
    assert len({p["source_dataset"] for p in picked}) >= 3
    assert len({(p["metadata"]["original_width"], p["metadata"]["original_height"])
                for p in picked}) >= 2


def test_items_missing_a_biased_score_are_dropped_from_both_sides():
    """A biased call that failed to parse must not shrink only the "after" column.
    Here the judge tracks the humans perfectly on every item it answered; the drop
    is confined to one item, so a correctly-paired comparison shows no change."""
    samples, base, biased = [], [], []
    for t in range(3):
        for m in range(4):
            sid = f"t{t}_m{m}"
            samples.append(_sample(sid, f"turn{t}", 0.1 * (m + 1)))
            base.append(_res(f"b::{sid}", sid, m + 1))
            if sid != "t0_m3":  # one biased call never came back
                biased.append(_res(f"x::{sid}", sid, m + 1, bias="padding"))
    st = compute_agreement(samples, base, biased, judge_model="j",
                           bias_type="padding", n_boot=50)
    assert st.n_items == 12 and st.n_paired == 11
    assert st.spearman_original == pytest.approx(st.spearman_biased)
    assert st.spearman_delta == pytest.approx(0.0)
    assert st.accuracy_original == st.accuracy_biased
    assert "n_paired" in st.as_row()


# --------------------------------------------------------------------------- #
# A judge whose collection is still in progress must not enter a published table.
# Found 2026-08-17 while designing WP-A5b's calibration batch: discovery is by glob,
# so a new judge appears the moment its FIRST row lands, and because BH runs per
# declared family, a 6th judge takes claim A's family from 15 cells to 18 and changes
# EVERY published q-value.  Nothing errors -- the numbers just stop matching the paper.
# --------------------------------------------------------------------------- #
def _judge_file(results_dir: Path, task: str, model: str, n: int) -> None:
    p = results_dir / "raw_judgments" / f"{task}__{model}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join('{"x":%d}' % i for i in range(n)) + "\n", encoding="utf-8")


def test_a_half_collected_judge_is_excluded_from_discovery(tmp_path: Path):
    from edit_judge_bias.experiments.build_claim_tables import _discover_models

    for m in ("gpt-5.5", "gemini-3.5-flash"):
        _judge_file(tmp_path, "scoring", m, 11_679)
    _judge_file(tmp_path, "scoring", "llama-4-scout", 84)   # the calibration batch

    assert _discover_models(tmp_path, "scoring") == ["gemini-3.5-flash", "gpt-5.5"]


def test_the_exclusion_is_recorded_not_silent(tmp_path: Path):
    from edit_judge_bias.experiments.build_claim_tables import (
        PARTIAL_JUDGE_LOG, _discover_models,
    )

    _judge_file(tmp_path, "scoring", "gpt-5.5", 1000)
    _judge_file(tmp_path, "scoring", "new-judge", 84)
    _discover_models(tmp_path, "scoring")

    log = json.loads((tmp_path / "metrics" / PARTIAL_JUDGE_LOG).read_text(encoding="utf-8"))
    assert log["scoring"]["excluded"] == {"new-judge": 84}
    assert log["scoring"]["included"] == ["gpt-5.5"]


def test_a_complete_new_judge_is_admitted(tmp_path: Path):
    """The filter must not become a permanent lock-out: A5b's whole point is a 6th judge."""
    from edit_judge_bias.experiments.build_claim_tables import _discover_models

    _judge_file(tmp_path, "scoring", "gpt-5.5", 11_679)
    _judge_file(tmp_path, "scoring", "llama-4-scout", 11_600)
    assert _discover_models(tmp_path, "scoring") == ["gpt-5.5", "llama-4-scout"]


def test_include_partial_escape_hatch_restores_the_old_behaviour(tmp_path: Path):
    from edit_judge_bias.experiments.build_claim_tables import _discover_models

    _judge_file(tmp_path, "scoring", "gpt-5.5", 1000)
    _judge_file(tmp_path, "scoring", "new-judge", 3)
    assert _discover_models(tmp_path, "scoring", include_partial=True) == [
        "gpt-5.5", "new-judge"]


def test_every_builder_that_globs_judges_uses_the_same_guard():
    """The guard had a SECOND DOOR, found 2026-08-17 before the calibration was paid for.

    It was written for `build_claim_tables` alone, but `aggregate_results` globs the same
    directory and feeds `scoring_shift.csv` / `scoring_shift_qc.csv` -- the legacy pair
    that QC-filtered effect sizes are still quoted from.  Two n=6 judge rows would
    have appeared there the moment the calibration landed, and the acceptance check
    ("33 CSVs unchanged after a full rebuild") would have failed on tables nobody watches.

    Pinned as an IDENTITY rather than by re-testing the behaviour: a copy-paste of the
    filter into the second module would pass a behavioural test and then drift.  The
    third builder, `build_analysis_tables`, is safe by a different mechanism and is
    asserted separately below, so that mechanism cannot be deleted silently either.
    """
    from edit_judge_bias.experiments import aggregate_results, build_claim_tables

    assert build_claim_tables._discover_models is aggregate_results._discover_models


def test_the_guard_survives_a_console_that_cannot_encode_its_own_marker(tmp_path, capsys,
                                                                        monkeypatch):
    """The guard died inside its own warning the first time it ever fired (2026-08-18).

    This project's Windows console is GBK and the house marker is "⚠️", which GBK cannot
    encode -- so `scripts/05_compute_metrics.sh` raised UnicodeEncodeError at exactly the
    moment the guard had something to say, and the rebuild stopped.  It had never been
    exercised because no partially-collected judge had existed until the A5b calibration.
    """
    import io as _io

    from edit_judge_bias.experiments import aggregate_results as ag

    _judge_file(tmp_path, "scoring", "gpt-5.5", 1000)
    _judge_file(tmp_path, "scoring", "llama-4-scout", 84)

    class GbkStdout(_io.StringIO):
        encoding = "gbk"

        def write(self, s):  # noqa: D102
            s.encode("gbk")  # raises exactly as the real console does
            return super().write(s)

    monkeypatch.setattr("sys.stdout", GbkStdout())
    assert ag._discover_models(tmp_path, "scoring") == ["gpt-5.5"]
    printed = sys.stdout.getvalue()
    assert "llama-4-scout n=84" in printed, "the payload must survive even if the glyph does not"


#: Modules allowed to glob `scoring__*.jsonl` directly, and why.  None: every builder
#: discovers judges through the guard.
_UNGUARDED_GLOB_OK: set = set()


def test_no_builder_rediscovers_judges_behind_the_guards_back():
    """The guard had a THIRD door, and it was the destructive one (2026-08-18).

    `build_claim_b_permutation.build_cells` globbed the judge files itself and then
    INTERSECTED the answered-item sets across judges.  So a judge with a handful of
    breadth rows did not add a row -- it emptied the table: `claim_b_permutation.csv`
    went from its 8 published rows to 0 (that is §5.5's p_selected 0.0305 / 1.0e-4)
    while printing "no anchor cells found -- nothing to permute", which reads like a
    remark about the data rather than a destroyed headline.

    A behavioural test cannot cover this cheaply -- each builder needs a different
    fixture -- so the invariant is asserted structurally: inside `experiments/`, judge
    discovery happens in exactly one place.
    """
    import re

    src = REPO / "src" / "edit_judge_bias" / "experiments"
    bare = re.compile(r'glob\(\s*f?["\']scoring__\*\.jsonl["\']')
    offenders = sorted(
        p.name for p in src.glob("*.py")
        if p.name not in _UNGUARDED_GLOB_OK and bare.search(p.read_text(encoding="utf-8"))
    )
    assert not offenders, (
        f"{offenders} discover judges by a bare glob instead of "
        "`aggregate_results._discover_models`; a partially-collected judge would reach "
        "a published table through them"
    )


def test_every_published_table_builder_defaults_to_the_published_roster():
    """Coverage and roster are different guards, and the second one was added late.

    Routing the builders through the shared coverage guard (2026-08-18) fixed the
    "a partial judge corrupts a table" failure but created a second one: coverage stops
    excluding a judge once it is complete, so `build_leaderboard_simulation` went from
    40 published rows to 48 the moment WP-A5's sixth judge finished its anchor arm.
    Every builder that writes a PUBLISHED table therefore has to default to the
    published roster, not merely to "whatever finished".
    """
    import inspect

    from edit_judge_bias.experiments import (
        build_claim_b_permutation as perm,
        build_claim_tables as bct,
        build_leaderboard_simulation as lead,
        build_sensitivity_tables as sens,
    )

    checks = [
        (bct.build_claim_a, "roster"), (bct.build_claim_b, "roster"),
        (bct.build_pairwise, "roster"), (bct.build_retest, "roster"),
        (bct.build_all, "roster"),
        (perm.build_cells, "roster"), (perm.build_permutation_rows, "roster"),
        (lead.build_boards, "roster"), (lead.build_leaderboard_rows, "roster"),
        (sens._discover_scoring_models, "roster"),
    ]
    for fn, param in checks:
        sig = inspect.signature(fn)
        assert param in sig.parameters, f"{fn.__module__}.{fn.__qualname__} has no roster"
        default = sig.parameters[param].default
        assert default is bct.PUBLISHED_ROSTER, (
            f"{fn.__module__}.{fn.__qualname__} defaults to {default!r}; it must default "
            "to PUBLISHED_ROSTER so a forgotten argument omits a judge rather than "
            "admitting one into a published family"
        )


def test_the_third_builder_is_fenced_by_an_explicit_roster():
    from edit_judge_bias.experiments.build_analysis_tables import ROSTER

    assert set(ROSTER) == {
        "gpt-5.5", "gemini-3.5-flash", "gpt-4o-viescore", "qwen3.5-plus", "kimi-k2.5",
    }, "the published five; a new judge joins these tables only by an explicit edit"


# --------------------------------------------------------------------------- #
# Coverage protects a table WHILE a judge is collected. It cannot express "this
# judge is never in that family", which is what the user ruled for WP-A5 on
# 2026-08-18 -- and coverage alone would admit qwen3-vl-32b-instruct the moment
# the retest arm pushes it past 50% of the leading judge's baseline rows.
# --------------------------------------------------------------------------- #
def test_a_complete_new_judge_still_stays_out_of_the_published_bh_family(tmp_path: Path):
    from edit_judge_bias.experiments.build_claim_tables import (
        PUBLISHED_ROSTER, _judges,
    )

    for m in PUBLISHED_ROSTER:
        _judge_file(tmp_path, "scoring", m, 1396)
    _judge_file(tmp_path, "scoring", "qwen3-vl-32b-instruct", 1396)  # fully collected

    assert "qwen3-vl-32b-instruct" not in _judges(tmp_path, "scoring", PUBLISHED_ROSTER)
    assert sorted(_judges(tmp_path, "scoring", PUBLISHED_ROSTER)) == sorted(PUBLISHED_ROSTER)


def test_the_a5_roster_sees_only_the_new_judges(tmp_path: Path):
    from edit_judge_bias.experiments.build_claim_tables import (
        A5_ROSTER, PUBLISHED_ROSTER, _judges,
    )

    for m in PUBLISHED_ROSTER:
        _judge_file(tmp_path, "scoring", m, 1396)
    _judge_file(tmp_path, "scoring", "qwen3-vl-32b-instruct", 1396)

    assert _judges(tmp_path, "scoring", A5_ROSTER) == ["qwen3-vl-32b-instruct"]


def test_a_dropped_candidate_is_in_no_roster_but_is_still_on_the_record():
    """WP-F4/P3.  `llama-4-scout` was evaluated and rejected; it stayed in `A5_ROSTER`
    for a day, invisible only because its 84 calibration rows are below the coverage
    guard's floor.  A roster is a declaration of what a table's family IS, and the
    coverage guard cannot substitute for it -- the guard answers "has this judge
    finished?", so it stops hiding the name the moment somebody finishes the arm.

    Both halves are pinned: the name is in no roster (so it can never widen a BH family),
    and the reason it was dropped is still in the module (so removing it from the roster
    did not also remove the record of why)."""
    from edit_judge_bias.experiments.aggregate_results import (
        CANDIDATE_JUDGES, ROSTERS,
    )

    for name, roster in ROSTERS.items():
        if roster is None:      # "all" is exploration-only and is allowed to see everything
            continue
        assert "llama-4-scout" not in roster, name
    assert "llama-4-scout" in CANDIDATE_JUDGES
    why = CANDIDATE_JUDGES["llama-4-scout"]
    assert "54%" in why and "2.4%" in why, "the two measured reasons must travel with it"


def test_candidate_judges_can_never_become_a_table():
    """It is a record, not a roster: if it were reachable through `ROSTERS` somebody could
    pass `--roster candidates` and build a paper-shaped table out of a rejected judge."""
    from edit_judge_bias.experiments.aggregate_results import CANDIDATE_JUDGES, ROSTERS

    for roster in ROSTERS.values():
        if roster is None:
            continue
        assert not (set(roster) & set(CANDIDATE_JUDGES))


def test_the_two_rosters_never_overlap():
    """One judge in two declared families would be corrected twice for the same test."""
    from edit_judge_bias.experiments.build_claim_tables import A5_ROSTER, PUBLISHED_ROSTER

    assert not (set(A5_ROSTER) & set(PUBLISHED_ROSTER))


def test_a_non_published_roster_cannot_overwrite_the_published_tables():
    """`--roster a5` must write a5_claim_a.csv, never claim_a.csv."""
    import inspect

    from edit_judge_bias.experiments import build_claim_tables as bct

    src = inspect.getsource(bct.main)
    assert 'prefix="" if args.roster == "published"' in src, (
        "a non-published roster must be written under its own prefix, or it would "
        "overwrite the very tables the roster exists to keep byte-stable"
    )
