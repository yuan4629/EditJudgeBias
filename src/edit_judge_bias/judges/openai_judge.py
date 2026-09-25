"""OpenAI-compatible MLLM judge adapter (Milestone 4).

Talks to any OpenAI-style `/chat/completions` endpoint (the OpenAI API or an
OpenAI-compatible gateway). Images are read locally and inlined as base64 data URLs. Network access is gated by the runner's `--use-api` flag and the key is read
from an environment variable — never hard-coded or stored in a config file.

Robustness: temperature defaults to 0,
transient HTTP errors are retried with exponential backoff, and identical requests
are cached on disk so a rerun never pays twice for the same (model, prompt, images).
Uses only the stdlib (urllib) — no extra dependency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_USAGE_LOCK = threading.Lock()

from edit_judge_bias.judges.base import JudgeAdapter, JudgeRequest

# 400 included: multi-backend gateways intermittently 400 on a backend that rejects a
# valid param (e.g. one backend lacks a reasoning_effort value) — a retry often
# routes to a working backend. Genuinely-permanent 400s are caught by _is_permanent.
_RETRYABLE_STATUS = {400, 408, 409, 429, 500, 502, 503, 504}
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

# Substrings marking an error that retrying will never fix (e.g. the gateway has no
# channel for the requested model). Matched against the response body; the Chinese
# strings are the equivalent messages returned by common self-hosted gateways.
_PERMANENT_MARKERS = (
    "无可用渠道",
    "no available channel",
    "无可用",
    "does not exist",
    "not found",
    "model_not_found",
    "invalid_api_key",
    "incorrect api key",
)


def _is_permanent(body: str) -> bool:
    low = body.lower()
    return any(m in body or m in low for m in _PERMANENT_MARKERS)


def _read_error_body(err: urllib.error.HTTPError) -> str:
    """Read an HTTPError's response body for diagnostics (best-effort)."""
    try:
        return err.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


class JudgeAPIError(RuntimeError):
    """Raised when a judge API call fails after exhausting retries."""


class OpenAICompatibleJudge(JudgeAdapter):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        *,
        api_model: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        image_detail: str = "auto",
        cache_dir: Optional[Path] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        usage_log: Optional[Path] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required (e.g. https://<relay-host>/v1)")
        if not api_key:
            raise ValueError("api_key is required (read from an environment variable)")
        self.model_name = model_name          # label stored on JudgeResult
        self.api_model = api_model or model_name  # actual model id sent to the API
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.image_detail = image_detail
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.extra_params = dict(extra_params or {})
        self.usage_log = Path(usage_log) if usage_log else None

    # ----------------------------------------------------------------- payload
    @staticmethod
    def _encode_image(path: Path) -> str:
        path = Path(path)
        mime = _MIME.get(path.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _build_payload(self, request: JudgeRequest) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for img in request.images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(img), "detail": self.image_detail},
            })
        payload: Dict[str, Any] = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.extra_params)
        return payload

    # ------------------------------------------------------------------- cache
    def _cache_key(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.model_name.replace("/", "_") / f"{key}.txt"

    # -------------------------------------------------------------------- http
    def _single_post(self, body: bytes) -> str:
        """One POST. Returns the response body text; raises on HTTP/URL error."""
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def _post_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return json.loads(self._single_post(body))
            except urllib.error.HTTPError as e:  # noqa: PERF203
                err_body = _read_error_body(e)
                last_err = e
                permanent = e.code not in _RETRYABLE_STATUS or _is_permanent(err_body)
                if permanent or attempt == self.max_retries:
                    raise JudgeAPIError(
                        f"HTTP {e.code} from judge API: {err_body[:300] or e}"
                    ) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                # ConnectionError covers RemoteDisconnected / reset — common on busy relays.
                last_err = e
                if attempt == self.max_retries:
                    raise JudgeAPIError(f"judge API call failed: {e}") from e
            time.sleep(self.backoff_base ** attempt)
        raise JudgeAPIError(f"judge API call failed: {last_err}")  # pragma: no cover

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeAPIError(f"unexpected response shape: {data!r}") from e

    def _log_usage(self, data: Dict[str, Any]) -> None:
        """Append the API's token usage to the usage log (for live cost tracking)."""
        if self.usage_log is None:
            return
        u = data.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        row = {
            "model": self.model_name,
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        }
        with _USAGE_LOCK:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- generate
    def generate(self, request: JudgeRequest) -> str:
        payload = self._build_payload(request)
        cache_path = self._cache_path(self._cache_key(payload))
        if cache_path is not None and cache_path.is_file():
            return cache_path.read_text(encoding="utf-8")
        data = self._post_with_retry(payload)
        self._log_usage(data)
        text = self._extract_text(data)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, config: Dict[str, Any], root: Optional[Path] = None) -> "OpenAICompatibleJudge":
        api_key_env = config.get("api_key_env", "EDITJUDGE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found: set environment variable {api_key_env!r}"
            )
        base_url = config.get("base_url") or os.environ.get("EDITJUDGE_BASE_URL", "")
        cache_dir = config.get("cache_dir")
        if cache_dir and root is not None:
            cache_dir = Path(root) / cache_dir
        usage_log = config.get("usage_log", "results/logs/usage.jsonl")
        if usage_log and root is not None:
            usage_log = Path(root) / usage_log
        return cls(
            model_name=config["model_name"],
            base_url=base_url,
            api_key=api_key,
            api_model=config.get("api_model"),
            temperature=config.get("temperature", 0.0),
            max_tokens=int(config.get("max_tokens", 1024)),
            timeout=float(config.get("timeout", 60.0)),
            max_retries=int(config.get("max_retries", 4)),
            image_detail=config.get("image_detail", "auto"),
            cache_dir=cache_dir,
            extra_params=config.get("extra_params"),
            usage_log=usage_log,
        )
