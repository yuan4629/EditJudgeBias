"""Tests for the fairness track (attribute templates + edit client), no network."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from edit_judge_bias.fairness.attribute_templates import (
    available_attributes,
    variants_for,
)
from edit_judge_bias.fairness.image_edit_client import (
    ImageEditError,
    QwenImageEditClient,
    _multipart,
)


def test_attribute_templates_cover_gender_and_skin_tone():
    assert set(available_attributes()) == {"gender", "skin_tone"}
    labels = {v.label for v in variants_for("gender")}
    assert labels == {"woman", "man"}
    # Each prompt must include the identity-preservation clause.
    for v in variants_for("skin_tone"):
        assert "exactly the same" in v.prompt or "the same" in v.prompt


def test_unknown_attribute_raises():
    with pytest.raises(KeyError):
        variants_for("height")


def test_multipart_encodes_fields_and_image():
    body, ctype = _multipart({"model": "qwen-image-edit", "prompt": "hi", "size": None},
                             b"\x89PNGDATA", "orig.png")
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body and b"qwen-image-edit" in body
    assert b'name="prompt"' in body and b"hi" in body
    assert b'filename="orig.png"' in body and b"\x89PNGDATA" in body
    assert b'name="size"' not in body  # None fields are skipped


def test_endpoint_url_normalization():
    assert QwenImageEditClient("https://x/v1")._endpoint() == "https://x/v1/images/edits"
    assert QwenImageEditClient("https://x")._endpoint() == "https://x/v1/images/edits"
    assert QwenImageEditClient("https://x/v1/images/edits")._endpoint() == "https://x/v1/images/edits"


def test_decode_b64_and_url_and_empty():
    client = QwenImageEditClient("https://x/v1")
    import base64

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = client._decode({"data": [{"b64_json": b64}]})
    assert img.size == (4, 4)
    with pytest.raises(ImageEditError):
        client._decode({"data": []})


def test_edit_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("EDITJUDGE_IMAGE_EDIT_KEY", raising=False)
    src = tmp_path / "o.png"
    Image.new("RGB", (8, 8)).save(src)
    with pytest.raises(ImageEditError, match="is empty"):
        QwenImageEditClient("https://x/v1").edit(src, "change the person to a woman")
