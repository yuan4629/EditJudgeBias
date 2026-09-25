"""Tests for the OpenAI-compatible judge adapter (Milestone 4). No real network."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest
from conftest import make_rgb_image

from edit_judge_bias.judges import build_adapter
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.openai_judge import JudgeAPIError, OpenAICompatibleJudge

_OK = json.dumps({"choices": [{"message": {"role": "assistant", "content": '{"winner": "A"}'}}]})


def _adapter(tmp_path: Path, **kw) -> OpenAICompatibleJudge:
    return OpenAICompatibleJudge(
        model_name="gpt-5.4-nano",
        base_url="https://relay.example/v1",
        api_key="sk-test",
        **kw,
    )


def _req(tmp_path: Path, task="scoring"):
    imgs = [make_rgb_image(tmp_path / "o.png"), make_rgb_image(tmp_path / "e.png")]
    return JudgeRequest(prompt="judge this", images=imgs, task=task)


def test_payload_structure_and_base64_images(tmp_path: Path):
    a = _adapter(tmp_path)
    payload = a._build_payload(_req(tmp_path))
    assert payload["model"] == "gpt-5.4-nano"
    assert payload["temperature"] == 0.0
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "judge this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_endpoint_built_from_base_url(tmp_path: Path):
    a = _adapter(tmp_path)
    assert a._endpoint == "https://relay.example/v1/chat/completions"


def test_api_model_decoupled_from_label(tmp_path: Path):
    a = OpenAICompatibleJudge(
        model_name="gpt-4o-viescore", api_model="gpt-4o",
        base_url="https://relay.example/v1", api_key="sk-test",
    )
    assert a.model_name == "gpt-4o-viescore"      # metrics label
    assert a._build_payload(_req(tmp_path))["model"] == "gpt-4o"  # actual API id


def test_temperature_omitted_when_none(tmp_path: Path):
    a = _adapter(tmp_path, temperature=None)
    assert "temperature" not in a._build_payload(_req(tmp_path))


def test_generate_parses_choices(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path)
    monkeypatch.setattr(a, "_single_post", lambda body: _OK)
    assert a.generate(_req(tmp_path)) == '{"winner": "A"}'


def test_usage_is_logged(tmp_path: Path, monkeypatch):
    log = tmp_path / "usage.jsonl"
    a = _adapter(tmp_path, usage_log=log)
    resp = json.dumps({
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 400, "completion_tokens": 900,
                  "completion_tokens_details": {"reasoning_tokens": 850}},
    })
    monkeypatch.setattr(a, "_single_post", lambda body: resp)
    a.generate(_req(tmp_path))
    row = json.loads(log.read_text(encoding="utf-8").strip())
    assert row["prompt_tokens"] == 400 and row["reasoning_tokens"] == 850


def test_caching_avoids_second_post(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path, cache_dir=tmp_path / "cache")
    calls = {"n": 0}

    def fake_post(body):
        calls["n"] += 1
        return _OK

    monkeypatch.setattr(a, "_single_post", fake_post)
    req = _req(tmp_path)
    a.generate(req)
    a.generate(req)  # identical -> served from cache
    assert calls["n"] == 1


def test_retries_then_succeeds(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path, max_retries=3, backoff_base=1.0)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError("u", 503, "busy", {}, None)
        return _OK

    monkeypatch.setattr(a, "_single_post", flaky)
    assert a.generate(_req(tmp_path)) == '{"winner": "A"}'
    assert calls["n"] == 3


def test_non_retryable_status_raises_immediately(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path, max_retries=5, backoff_base=1.0)
    calls = {"n": 0}

    def bad(body):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "unauthorized", {}, None)

    monkeypatch.setattr(a, "_single_post", bad)
    with pytest.raises(JudgeAPIError):
        a.generate(_req(tmp_path))
    assert calls["n"] == 1  # 401 (auth) is not retried


def test_intermittent_400_is_retried(tmp_path: Path, monkeypatch):
    # The relay sometimes 400s on one backend; a retry can route to a working one.
    a = _adapter(tmp_path, max_retries=3, backoff_base=1.0)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.HTTPError("u", 400, "bad backend", {}, None)
        return _OK

    monkeypatch.setattr(a, "_single_post", flaky)
    assert a.generate(_req(tmp_path)) == '{"winner": "A"}'
    assert calls["n"] == 2


def test_retry_exhaustion_raises(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path, max_retries=2, backoff_base=1.0)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(a, "_single_post", lambda body: (_ for _ in ()).throw(
        urllib.error.URLError("conn refused")))
    with pytest.raises(JudgeAPIError):
        a.generate(_req(tmp_path))


def test_permanent_503_no_channel_is_not_retried(tmp_path: Path, monkeypatch):
    # A 503 whose body says "no available channel" must fail fast (no retries).
    a = _adapter(tmp_path, max_retries=5, backoff_base=1.0)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    class _Body:
        def read(self_inner):
            return "无可用渠道，请联系管理员".encode("utf-8")

    def no_channel(body):
        calls["n"] += 1
        err = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
        err.read = _Body().read  # attach a readable body
        raise err

    monkeypatch.setattr(a, "_single_post", no_channel)
    with pytest.raises(JudgeAPIError, match="无可用渠道"):
        a.generate(_req(tmp_path))
    assert calls["n"] == 1  # not retried despite 503


def test_unexpected_response_shape_raises(tmp_path: Path, monkeypatch):
    a = _adapter(tmp_path)
    monkeypatch.setattr(a, "_single_post", lambda body: json.dumps({"oops": 1}))
    with pytest.raises(JudgeAPIError):
        a.generate(_req(tmp_path))


# --------------------------------------------------------------------------- #
# build_adapter gating + from_config                                          #
# --------------------------------------------------------------------------- #
def test_build_adapter_requires_use_api():
    with pytest.raises(PermissionError):
        build_adapter({"type": "openai", "model_name": "x"}, use_api=False)


def test_from_config_reads_key_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITJUDGE_API_KEY", "sk-zzz")
    a = build_adapter(
        {"type": "openai", "model_name": "gpt-5.4-nano",
         "base_url": "https://r/v1", "api_key_env": "EDITJUDGE_API_KEY"},
        use_api=True,
    )
    assert isinstance(a, OpenAICompatibleJudge)
    assert a._api_key == "sk-zzz"


def test_from_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("EDITJUDGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key not found"):
        build_adapter(
            {"type": "openai", "model_name": "x", "base_url": "https://r/v1"},
            use_api=True,
        )
