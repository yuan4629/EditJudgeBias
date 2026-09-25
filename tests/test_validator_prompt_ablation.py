"""WP-A4c — the C-class ablation variant of the quality-preservation validator prompt.

The published template names four examples of "cosmetic" artefacts and every one is
B-class (global brightness/saturation, borders, watermarks, edge text bands); the four
C-class overlays are never named.  That asymmetry was a footnote while no cue sat
significantly below its floor.  After WP-A3 it is load-bearing: `zoom_inset` is now
significant on two validators, and "the validator saw the inset" is observationally
equivalent to "the edit was destroyed" until this ablation separates them.

So the variant has to differ from the published template in EXACTLY the intended places
and nowhere else -- otherwise a pass-rate difference is no longer attributable to the
asymmetry being tested.
"""

from __future__ import annotations

import pytest

from edit_judge_bias.prompts.validator_prompt import (
    STYLES,
    _ABLATION_TEMPLATE,
    _TEMPLATE,
    build_validator_prompt,
)


def test_default_style_is_byte_identical_to_the_published_template():
    """The published grid was collected under this exact string. If it moves, every
    published pass rate silently stops being reproducible."""
    assert build_validator_prompt("Remove the cat") == _TEMPLATE.format(
        instruction="Remove the cat"
    )
    assert build_validator_prompt("Remove the cat", "default") == build_validator_prompt(
        "Remove the cat"
    )


def test_the_ablation_differs_in_exactly_two_places():
    a = build_validator_prompt("x", "default").splitlines()
    b = build_validator_prompt("x", "c_class_named").splitlines()
    assert len(a) == len(b), "the variant must not add or remove lines"
    differing = [i for i, (la, lb) in enumerate(zip(a, b)) if la != lb]
    assert len(differing) == 2, f"expected 2 changed lines, got {differing}"


def test_the_ablation_names_all_four_c_class_overlays():
    text = build_validator_prompt("x", "c_class_named").lower()
    for phrase in ("box or arrow", "magnified inset", "caption bar", "sticker"):
        assert phrase in text, phrase


def test_the_ablation_keeps_the_b_class_examples():
    """It ADDS C-class to the cosmetic list; it must not remove B-class, or the contrast
    becomes 'which class is named' rather than 'is C-class named too'."""
    text = build_validator_prompt("x", "c_class_named")
    for phrase in ("brightness/saturation", "borders", "watermarks", "edge text bands"):
        assert phrase in text, phrase


def test_the_json_contract_is_untouched():
    """A different output schema would make the two arms unparseable by one parser and
    the comparison meaningless."""
    a, b = (build_validator_prompt("x", s) for s in ("default", "c_class_named"))
    for key in ("instruction_adherence_changed", "editing_quality_changed",
                "detail_preservation_changed", "major_semantic_shift", '"pass"'):
        assert key in a and key in b


def test_an_unknown_style_raises_rather_than_falling_back():
    """A typo in a config must not silently collect a whole paid arm under the published
    prompt and label it an ablation."""
    with pytest.raises(ValueError, match="unknown validator prompt style"):
        build_validator_prompt("x", "c_class")          # plausible typo
    assert set(STYLES) == {"default", "c_class_named"}


def test_the_variant_was_actually_derived_not_silently_no_opped():
    """`str.replace` returns the string unchanged when the pattern misses, and the first
    draft of this variant did exactly that: the published template uses backslash line
    continuations, so a pattern written with the newlines visible in the source matched
    nothing and only half the ablation happened. `_sub_once` now raises instead."""
    assert _ABLATION_TEMPLATE != _TEMPLATE
    assert "annotation overlays" in _ABLATION_TEMPLATE


def test_the_runner_defaults_to_the_published_prompt():
    """Every existing validator config omits `prompt_style:`; they must keep collecting
    under the published template."""
    import inspect

    from edit_judge_bias.experiments import run_quality_validation

    src = inspect.getsource(run_quality_validation.run)
    assert 'cfg.get("prompt_style", "default")' in src
