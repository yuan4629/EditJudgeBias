"""Regression tests for the D-class fairness pipeline.

Each test here pins a failure mode that would have produced a publishable-looking wrong
number, in the same spirit as tests/test_grid_isolation.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import (
    ContentCategory,
    EditType,
    JudgeResult,
    SampleRecord,
)
from edit_judge_bias.experiments import build_fairness_samples, run_counterfactual_edit
from edit_judge_bias.experiments.run_attribute_edit import _edit_id, _eligible
from edit_judge_bias.experiments.run_attribute_validation import (
    build_pairs,
    passed_pair_keys,
)
from edit_judge_bias.fairness.records import (
    AttributeEditRecord,
    AttributeValidationResult,
    CounterfactualEditRecord,
)
from edit_judge_bias.judges.parser import parse_attribute_validation
from edit_judge_bias.metrics.fairness_metrics import (
    keys_from_manifest,
    compute_attribute_gaps,
    compute_editor_noise,
    minimum_detectable_effect,
)


# --------------------------------------------------------------------------- #
# eligibility screen                                                          #
# --------------------------------------------------------------------------- #
def _sample(sample_id: str, instruction: str, category=ContentCategory.HUMAN) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_dataset="toy",
        edit_type=EditType.ADD,
        content_category=category,
        original_image_path="a.png",
        instruction=instruction,
        edit_model="m",
        edited_image_path="b.png",
    )


def test_eligible_requires_human_category():
    assert _eligible(_sample("s1", "add a hat"))
    assert not _eligible(_sample("s2", "add a hat", ContentCategory.OBJECT))


@pytest.mark.parametrize("instruction", [
    "make the woman smile",          # attribute word: the instruction depends on the attribute
    "change his hat to red",         # gendered pronoun
    "give her longer hair",          # attribute + hair
])
def test_eligible_rejects_attribute_referencing_instructions(instruction):
    assert not _eligible(_sample("s", instruction))


@pytest.mark.parametrize("instruction", [
    "change the girl's shirt to red",
    "add a toy next to the child",
    "make the baby laugh",
])
def test_eligible_rejects_minors_as_an_ethics_hard_stop(instruction):
    """Not a coverage tradeoff — this screen must never be relaxed to grow the pool."""
    assert not _eligible(_sample("s", instruction))


def test_edit_id_is_stable_and_unique_per_variant():
    assert _edit_id("s1", "gender", "woman") == "s1__gender__woman"
    assert _edit_id("s1", "gender", "woman") != _edit_id("s1", "gender", "man")


# --------------------------------------------------------------------------- #
# ★ the cache-off guard on the editor re-render null control                   #
# --------------------------------------------------------------------------- #
def test_repeat_gt_1_with_a_cache_dir_raises(tmp_path: Path):
    """A cached second render returns the first image byte for byte.

    That would report an editor-noise floor of exactly zero — a fabricated null, and the null
    is the only thing an attribute gap can be read against. Same trap the scoring retest arm
    hit; this is its D-class twin.
    """
    cfg = {
        "base_url": "https://example.invalid/v1",
        "attribute_edits": "data/manifests/attribute_edits.jsonl",
        "output_dir": "out",
        "manifest_out": "m.jsonl",
        "repeat": 2,
        "cache_dir": "results/cache/image_edit",
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="cached second render"):
        run_counterfactual_edit.run(p, root=tmp_path, dry_run=True)


def test_repeat_gt_1_is_allowed_once_the_cache_is_off(tmp_path: Path):
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    io.write_jsonl(tmp_path / "data/manifests/attribute_edits.jsonl", [])
    cfg = {
        "base_url": "https://example.invalid/v1",
        "attribute_edits": "data/manifests/attribute_edits.jsonl",
        "output_dir": "out",
        "manifest_out": "m.jsonl",
        "repeat": 2,
        "cache_dir": None,
    }
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run_counterfactual_edit.run(p, root=tmp_path, dry_run=True)["repeat"] == 2


def test_render_id_leaves_the_first_render_unsuffixed():
    """Adding the null control later must not renumber renders already paid for."""
    assert run_counterfactual_edit._render_id("e1", 1) == "e1"
    assert run_counterfactual_edit._render_id("e1", 2) == "e1::render2"


# --------------------------------------------------------------------------- #
# the validation gate                                                         #
# --------------------------------------------------------------------------- #
def _edit(base: str, attribute: str, label: str) -> AttributeEditRecord:
    return AttributeEditRecord(
        edit_id=_edit_id(base, attribute, label),
        base_sample_id=base,
        attribute=attribute,
        variant_label=label,
        prompt="p",
        original_image_path="o.png",
        counterfactual_image_path=f"cf_{base}_{attribute}_{label}.png",
        edit_model="m",
        content_category="human",
        instruction="add a hat",
    )


def test_build_pairs_counts_single_variant_groups_as_incomplete():
    edits = [
        _edit("s1", "gender", "woman"), _edit("s1", "gender", "man"),
        _edit("s2", "gender", "woman"),                      # sibling refused
    ]
    pairs, incomplete = build_pairs(edits)
    assert [p[0] for p in pairs] == ["s1"]
    assert incomplete == 1


def test_parse_attribute_validation_derives_pass_strictly():
    """A missing field cannot count as satisfied.

    Two of the three conditions must be positively observed (a legible person, an actual flip),
    so treating an absent field as True would silently admit unusable pairs.
    """
    ok = parse_attribute_validation(
        '{"person_legible": true, "attribute_flipped": true, "scene_preserved": true}'
    )
    assert ok.success and ok.passed is True
    partial = parse_attribute_validation('{"person_legible": true, "scene_preserved": true}')
    assert partial.success and partial.passed is False


def test_passed_pair_keys_raises_when_the_validator_rubber_stamps(tmp_path: Path):
    """If the auditor 'finds' flips between identical images, its verdicts are worthless.

    Returning a pass set anyway would hand the study a filter that filtered nothing while
    looking like it had.
    """
    path = tmp_path / "val.jsonl"
    rows = [
        AttributeValidationResult(
            pair_key="s1__gender", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="woman", passed=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True,
        ),
        AttributeValidationResult(
            pair_key="s1__gender__CONTROL", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="man", is_control=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True, passed=True,
        ),
    ]
    io.write_jsonl(path, rows)
    with pytest.raises(ValueError, match="false-flip floor"):
        passed_pair_keys(path)


def test_two_auditors_gate_by_intersection_never_union(tmp_path: Path):
    """Adding scrutiny must SHRINK the admitted set, never grow it.

    The set means "pairs shown to be usable", so a union would make it grow as auditors were
    added — the exact direction error `aggregate_results._passed_biased_ids` made with the QC
    subset. The asymmetry is deliberate: an admitted-but-bad pair corrupts the gap, a
    dropped-but-good pair only costs power.
    """
    def _write(name, passing):
        rows = []
        for key in ("s1__gender", "s2__gender", "s3__gender"):
            rows.append(AttributeValidationResult(
                pair_key=key, base_sample_id=key.split("__")[0], attribute="gender",
                label_a="man", label_b="woman", passed=key in passing,
                person_legible=True, attribute_flipped=key in passing, scene_preserved=True,
            ))
            rows.append(AttributeValidationResult(
                pair_key=f"{key}__CONTROL", base_sample_id=key.split("__")[0],
                attribute="gender", label_a="man", label_b="man", is_control=True,
                person_legible=True, attribute_flipped=False, scene_preserved=True, passed=False,
            ))
        p = tmp_path / name
        io.write_jsonl(p, rows)
        return p

    a = _write("a.jsonl", {"s1__gender", "s2__gender"})
    b = _write("b.jsonl", {"s2__gender", "s3__gender"})
    assert passed_pair_keys(a) == {"s1__gender", "s2__gender"}
    assert passed_pair_keys([a, b]) == {"s2__gender"}          # intersection, not union
    assert len(passed_pair_keys([a, b])) < len(passed_pair_keys(a))


def test_a_rubber_stamping_auditor_raises_even_when_combined(tmp_path: Path):
    """One bad auditor must not be laundered by pairing it with a good one."""
    good = tmp_path / "good.jsonl"
    io.write_jsonl(good, [
        AttributeValidationResult(
            pair_key="s1__gender", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="woman", passed=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True),
        AttributeValidationResult(
            pair_key="s1__gender__CONTROL", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="man", is_control=True,
            person_legible=True, attribute_flipped=False, scene_preserved=True, passed=False),
    ])
    stamp = tmp_path / "stamp.jsonl"
    io.write_jsonl(stamp, [
        AttributeValidationResult(
            pair_key="s1__gender", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="woman", passed=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True),
        AttributeValidationResult(
            pair_key="s1__gender__CONTROL", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="man", is_control=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True, passed=True),
    ])
    with pytest.raises(ValueError, match="false-flip floor"):
        passed_pair_keys([good, stamp])


def test_passed_pair_keys_excludes_control_rows_from_the_pass_set(tmp_path: Path):
    path = tmp_path / "val.jsonl"
    io.write_jsonl(path, [
        AttributeValidationResult(
            pair_key="s1__gender", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="woman", passed=True,
            person_legible=True, attribute_flipped=True, scene_preserved=True,
        ),
        AttributeValidationResult(
            pair_key="s1__gender__CONTROL", base_sample_id="s1", attribute="gender",
            label_a="man", label_b="man", is_control=True,
            person_legible=True, attribute_flipped=False, scene_preserved=True, passed=False,
        ),
        AttributeValidationResult(
            pair_key="s2__gender", base_sample_id="s2", attribute="gender",
            label_a="man", label_b="woman", passed=False,
            person_legible=True, attribute_flipped=False, scene_preserved=True,
        ),
    ])
    assert passed_pair_keys(path) == {"s1__gender"}


# --------------------------------------------------------------------------- #
# ★ the gate must be applied PAIRWISE                                         #
# --------------------------------------------------------------------------- #
def _render(base: str, attribute: str, label: str, idx: int = 1) -> CounterfactualEditRecord:
    rid = _edit_id(base, attribute, label) + ("" if idx == 1 else f"::render{idx}")
    return CounterfactualEditRecord(
        render_id=rid,
        edit_id=_edit_id(base, attribute, label),
        base_sample_id=base,
        attribute=attribute,
        variant_label=label,
        render_index=idx,
        instruction="add a hat",
        counterfactual_image_path="cf.png",
        edited_image_path=f"ed_{rid}.png",
    )


def _fairness_fixture(tmp_path: Path, *, gate_pass=("s1__gender",)) -> Path:
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    (tmp_path / "results" / "v2_fairness" / "quality").mkdir(parents=True)
    io.write_jsonl(tmp_path / "data/manifests/samples_fairness_v2.jsonl",
                   [_sample("s1", "add a hat"), _sample("s2", "add a hat")])
    io.write_jsonl(tmp_path / "data/manifests/counterfactual_edits.jsonl", [
        _render("s1", "gender", "woman"), _render("s1", "gender", "man"),
        _render("s2", "gender", "woman"), _render("s2", "gender", "man"),
    ])
    rows = []
    for base in ("s1", "s2"):
        rows.append(AttributeValidationResult(
            pair_key=f"{base}__gender", base_sample_id=base, attribute="gender",
            label_a="man", label_b="woman",
            passed=f"{base}__gender" in gate_pass,
            person_legible=True, attribute_flipped=True, scene_preserved=True,
        ))
        rows.append(AttributeValidationResult(
            pair_key=f"{base}__gender__CONTROL", base_sample_id=base, attribute="gender",
            label_a="man", label_b="man", is_control=True,
            person_legible=True, attribute_flipped=False, scene_preserved=True, passed=False,
        ))
    io.write_jsonl(
        tmp_path / "results/v2_fairness/quality/attribute_validation__gpt-4o-mini.jsonl", rows
    )
    return tmp_path


def test_failed_gate_drops_both_variants_not_one(tmp_path: Path):
    """A pair that failed preservation is unusable from BOTH sides.

    Emitting the surviving variant alone would leave an orphan that the paired metric drops
    silently later, so the loss would never appear in any count.
    """
    root = _fairness_fixture(tmp_path, gate_pass=("s1__gender",))
    rep = build_fairness_samples.build(root=root, write=False)
    assert rep["samples_out"] == 2                      # s1's two variants only
    assert rep["dropped"]["failed_preservation_gate"] == 2


def test_incomplete_variant_pair_is_dropped_and_counted(tmp_path: Path):
    root = _fairness_fixture(tmp_path, gate_pass=("s1__gender", "s2__gender"))
    edits = io.read_jsonl(root / "data/manifests/counterfactual_edits.jsonl",
                          CounterfactualEditRecord)
    io.write_jsonl(root / "data/manifests/counterfactual_edits.jsonl",
                   [e for e in edits if e.render_id != "s2__gender__man"])
    rep = build_fairness_samples.build(root=root, write=False)
    assert rep["dropped"]["incomplete_variant_pair"] == 1
    assert rep["samples_out"] == 2


def test_null_control_renders_are_exempt_from_the_pair_requirement(tmp_path: Path):
    """The floor is deliberately single-variant; requiring a pair would delete it."""
    root = _fairness_fixture(tmp_path, gate_pass=("s1__gender",))
    edits = io.read_jsonl(root / "data/manifests/counterfactual_edits.jsonl",
                          CounterfactualEditRecord)
    io.write_jsonl(root / "data/manifests/counterfactual_edits.jsonl",
                   edits + [_render("s1", "gender", "woman", idx=2)])
    rep = build_fairness_samples.build(root=root, write=False)
    assert rep["null_control_renders"] == 1
    assert rep["study_renders"] == 2


# --------------------------------------------------------------------------- #
# the gap metric                                                              #
# --------------------------------------------------------------------------- #
#: (JudgeResult, SampleRecord) for one judged render. The pairing metadata goes on the SAMPLE,
#: never on the result — JudgeResult has no metadata container and forbids extra fields, so a
#: fixture that stashed `attribute` on the result would be testing an impossible layout.
def _result(judge: str, sample_id: str, score: int, *, base, attribute, label, idx=1):
    r = JudgeResult(
        result_id=f"score::{judge}::vanilla::{sample_id}",
        judge_model=judge,
        task_type="scoring",
        prompt_type="vanilla_scoring",
        raw_response_path=f"raw/{sample_id}.txt",
        sample_id=sample_id,
        overall_score=score,
        instruction_adherence=score,
        editing_quality=score,
        detail_preservation=score,
        fine_score=score * 3,
        score_scale=10,
        parse_success=True,
    )
    s = _sample(sample_id, "add a hat")
    s.metadata.base_sample_id = base
    s.metadata.attribute = attribute
    s.metadata.variant_label = label
    s.metadata.render_index = idx
    return r, s


def _gaps(*pairs, **kw):
    """compute_attribute_gaps over (result, sample) fixture pairs."""
    results = [p[0] for p in pairs]
    keys = keys_from_manifest([p[1] for p in pairs])
    return compute_attribute_gaps(results, keys, **kw)


def test_gap_is_standardised_by_the_paired_difference_sd_not_the_score_sd():
    """★ The units the reported MDE is in, pinned.

    `minimum_detectable_effect` is `(z_.975 + z_.80)/sqrt(n)` — SD of the *difference*. The
    write-up once standardised the observed gap by SD(*score*) instead and reported the D-S
    null as "4-8x below the MDE" when the consistent figure is 2.4-3.9x. Here the two
    denominators are made deliberately far apart: the scenes span a wide score range while
    every within-scene difference is ~1 point.
    """
    pairs = []
    for i, level in enumerate([1, 3, 5, 7, 9]):
        pairs += [
            _result("j", f"s{i}__gender__man", level + 1, base=f"s{i}",
                    attribute="gender", label="man"),
            _result("j", f"s{i}__gender__woman", level, base=f"s{i}",
                    attribute="gender", label="woman"),
        ]
    (stat,) = _gaps(*pairs)

    # fine_score = 3 x score, so every scene's gap is exactly 3 and the scores span 3..30.
    assert stat.mean_gap == pytest.approx(3.0)
    assert stat.sd_gap == pytest.approx(0.0, abs=1e-9)      # differences are constant
    row = stat.as_row()
    assert row["sd_gap"] == pytest.approx(0.0)
    # Guarded, not a ZeroDivisionError, and NOT silently falling back to a score SD.
    assert row["gap_in_sd_units"] is None


def test_gap_in_sd_units_divides_by_sd_gap():
    (stat,) = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s2__gender__man", 5, base="s2", attribute="gender", label="man"),
        _result("j", "s2__gender__woman", 7, base="s2", attribute="gender", label="woman"),
        _result("j", "s3__gender__man", 9, base="s3", attribute="gender", label="man"),
        _result("j", "s3__gender__woman", 6, base="s3", attribute="gender", label="woman"),
    )
    import statistics as _st
    gaps = [6.0, -6.0, 9.0]           # fine_score = 3 x score
    assert stat.sd_gap == pytest.approx(_st.stdev(gaps))
    assert stat.as_row()["gap_in_sd_units"] == pytest.approx(
        stat.mean_gap / _st.stdev(gaps), rel=1e-4)


def test_gap_is_paired_within_scene_and_signed_by_label_order():
    """man - woman, consistently, so the sign means one thing across every row."""
    (stat,) = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s2__gender__man", 5, base="s2", attribute="gender", label="man"),
        _result("j", "s2__gender__woman", 7, base="s2", attribute="gender", label="woman"),
    )
    assert (stat.label_a, stat.label_b) == ("man", "woman")
    assert stat.n == 2
    assert stat.mean_gap == pytest.approx(0.0)        # +6 and -6 in fine_score
    assert stat.mean_abs_gap == pytest.approx(6.0)    # ...which mean_abs_gap keeps visible


def test_scene_missing_one_variant_is_dropped_not_half_counted():
    (stat,) = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s2__gender__man", 5, base="s2", attribute="gender", label="man"),
    )
    assert stat.n == 1


def test_attributes_are_reported_separately_never_pooled():
    """User decision 2026-07-30: pooling would assert the two gaps are one quantity."""
    stats = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s1__skin_tone__dark", 4, base="s1", attribute="skin_tone", label="dark"),
        _result("j", "s1__skin_tone__light", 9, base="s1", attribute="skin_tone", label="light"),
    )
    assert sorted(s.attribute for s in stats) == ["gender", "skin_tone"]
    assert all(s.n == 1 for s in stats)


def test_null_control_renders_do_not_enter_the_gap():
    """A render-2 row shares (base, attribute, variant) with the study render.

    A join that ignored `render_index` would put two scores for the same variant into one cell
    and report editor noise as an attribute gap.
    """
    (stat,) = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s1__gender__woman__r2", 2, base="s1", attribute="gender",
                label="woman", idx=2),
    )
    assert stat.n == 1
    assert stat.mean_gap == pytest.approx(6.0)       # 24 - 18, the render-2 row ignored


def test_editor_noise_pairs_only_the_same_variant():
    pairs = [
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
        _result("j", "s1__gender__woman__r2", 8, base="s1", attribute="gender",
                label="woman", idx=2),
        _result("j", "s1__gender__man", 3, base="s1", attribute="gender", label="man"),
    ]
    noise = compute_editor_noise(
        [p[0] for p in pairs], keys_from_manifest([p[1] for p in pairs]),
        score_field="fine_score",
    )
    assert noise["j"] == (pytest.approx(6.0), 1)     # |24-18|, the `man` row not involved


def test_gap_carries_the_editor_noise_floor_for_comparison():
    (stat,) = _gaps(
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 7, base="s1", attribute="gender", label="woman"),
        _result("j", "s1__gender__woman__r2", 4, base="s1", attribute="gender",
                label="woman", idx=2),
    )
    assert stat.n_noise == 1
    assert stat.editor_noise_mean_abs == pytest.approx(9.0)
    # |gap| of 3 does not clear a floor of 9 — the row must say so rather than look significant.
    assert stat.as_row()["exceeds_editor_noise"] is False


def test_unkeyed_results_yield_no_rows_rather_than_wrong_rows():
    """If the manifest join produces nothing, the metric must be empty, not guess from ids."""
    pairs = [
        _result("j", "s1__gender__man", 8, base="s1", attribute="gender", label="man"),
        _result("j", "s1__gender__woman", 6, base="s1", attribute="gender", label="woman"),
    ]
    assert compute_attribute_gaps([p[0] for p in pairs], {}) == []


# --------------------------------------------------------------------------- #
# ★ one bad item must never abort the batch                                    #
# --------------------------------------------------------------------------- #
def test_connection_errors_are_classified_retryable():
    """`RemoteDisconnected` must be caught by the client's retry tuple.

    Measured 2026-07-30: it escaped the client, sailed past the runner's
    `except ImageEditError`, and killed a 164-render batch at render 61. It subclasses
    ConnectionResetError but NOT URLError, which is why a (URLError, TimeoutError) tuple
    could not see it.
    """
    import http.client

    from edit_judge_bias.fairness.image_edit_client import _CONNECTION_ERRORS

    assert issubclass(http.client.RemoteDisconnected, _CONNECTION_ERRORS)
    assert issubclass(http.client.IncompleteRead, _CONNECTION_ERRORS)
    assert issubclass(TimeoutError, _CONNECTION_ERRORS)
    # ...but a plain bug must still surface rather than be retried and mislabelled.
    assert not issubclass(ValueError, _CONNECTION_ERRORS)


def test_one_failing_render_is_logged_and_the_batch_continues(tmp_path: Path, monkeypatch):
    """The whole point of resume-by-id: a logged failure is cheap, a dead batch is not."""
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    edits = [_edit("s1", "gender", "woman"), _edit("s2", "gender", "man")]
    io.write_jsonl(tmp_path / "data/manifests/attribute_edits.jsonl", edits)
    for e in edits:
        p = tmp_path / e.counterfactual_image_path
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(p)

    calls = {"n": 0}

    class Flaky:
        def edit(self, path, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("Remote end closed connection without response")
            return Image.new("RGB", (8, 8))

    monkeypatch.setattr(run_counterfactual_edit, "QwenImageEditClient", lambda **kw: Flaky())
    cfg = {
        "base_url": "https://example.invalid/v1",
        "attribute_edits": "data/manifests/attribute_edits.jsonl",
        "output_dir": "out",
        "manifest_out": "data/manifests/counterfactual_edits.jsonl",
        "failure_log": "logs/fail.jsonl",
        "repeat": 1,
        "cache_dir": None,
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    stats = run_counterfactual_edit.run(p, root=tmp_path, use_api=True)
    assert stats == {"written": 1, "skipped": 0, "failed": 1}
    logged = [json.loads(l) for l in (tmp_path / "logs/fail.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(logged) == 1 and "ConnectionResetError" in logged[0]["message"]


def test_mde_scales_as_one_over_sqrt_n():
    """One number, derived one way, so the paper cannot quote two different MDEs."""
    assert minimum_detectable_effect(611) == pytest.approx(0.113)
    assert minimum_detectable_effect(41) == pytest.approx(0.436, abs=1e-3)
    assert minimum_detectable_effect(0) is None
