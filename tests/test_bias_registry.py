"""Tests for the bias registry (Milestone 2)."""

from __future__ import annotations

import pytest

from edit_judge_bias.bias.base import BiasInjector
from edit_judge_bias.bias.registry import available, get_injector, register
from edit_judge_bias.bias.text_overlay import TextOverlayInjector
from edit_judge_bias.bias.watermark import WatermarkInjector


def test_available_lists_the_five_mvp_injectors():
    assert set(available()) >= {
        "brightness",
        "saturation",
        "watermark",
        "text_overlay",
        "padding",
    }


def test_available_includes_full_version_injectors():
    assert set(available()) >= {
        "aesthetic_filter",   # B6
        "region_annotation",  # C1
        "zoom_inset",         # C2
        "detail_caption",     # C3
        "distraction",        # C4
    }


def test_c_class_injectors_declare_needs_original():
    for bt in ("region_annotation", "zoom_inset", "distraction"):
        assert get_injector(bt).needs_original is True
    # B6 and C3 work without the original image.
    assert get_injector("aesthetic_filter").needs_original is False
    assert get_injector("detail_caption").needs_original is False


def test_get_injector_returns_instance():
    inj = get_injector("brightness")
    assert isinstance(inj, BiasInjector)
    assert inj.bias_type == "brightness"


def test_aliases_resolve():
    assert isinstance(get_injector("instruction_overlay"), TextOverlayInjector)
    assert isinstance(get_injector("keyword_overlay"), TextOverlayInjector)
    assert isinstance(get_injector("authenticity_overlay"), WatermarkInjector)


def test_unknown_bias_type_raises():
    with pytest.raises(KeyError, match="unknown bias_type"):
        get_injector("teleport")


def test_position_is_a_protocol_bias_not_an_injector():
    with pytest.raises(KeyError, match="judge-protocol"):
        get_injector("position")


def test_register_custom_injector():
    class DummyInjector(BiasInjector):
        bias_type = "dummy_test_only"

        def transform(self, image, config, rng, sample):
            return image, {"bias_strength": 0.0}

    register(DummyInjector)
    try:
        assert isinstance(get_injector("dummy_test_only"), DummyInjector)
    finally:
        from edit_judge_bias.bias import registry

        registry._REGISTRY.pop("dummy_test_only", None)


def test_register_rejects_empty_bias_type():
    class Bad(BiasInjector):
        bias_type = ""

        def transform(self, image, config, rng, sample):
            return image, {}

    with pytest.raises(ValueError):
        register(Bad)
