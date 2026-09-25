"""Client for an OpenAI-compatible ``/v1/images/edits`` endpoint (fairness track).

Matches the user's Qwen-Image-Edit platform spec: multipart/form-data with
``model`` + ``prompt`` + a single ``image`` (binary), Bearer auth, and a Dall-e
style JSON response (``{"data": [{"b64_json": ...}]}`` or ``{"url": ...}``).

Design mirrors judges/openai_judge.py: stdlib-only (urllib), temperature-free,
retry with exponential backoff, and a disk cache keyed by (model, prompt, image
bytes, size) so a rerun never re-pays for the same counterfactual. Network use is
gated by the runner's ``--use-api`` flag; the key is read from an environment
variable, never hard-coded.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

#: Transport-level failures worth another attempt. Kept explicit (rather than a bare
#: `except Exception`) so a genuine bug still surfaces instead of being retried four times
#: and then reported as a network problem.
_CONNECTION_ERRORS = (
    urllib.error.URLError,          # DNS, refused, TLS
    TimeoutError,                   # socket.timeout is an alias in 3.10+
    ConnectionError,                # covers http.client.RemoteDisconnected
    http.client.HTTPException,      # BadStatusLine, IncompleteRead, ...
)
_PERMANENT_MARKERS = (
    "无可用渠道", "no available channel", "does not exist", "not found",
    "model_not_found", "invalid_api_key", "incorrect api key", "content_policy",
)


class ImageEditError(RuntimeError):
    """Raised when the edit endpoint fails permanently (after retries)."""


def _is_permanent(body: str) -> bool:
    low = body.lower()
    return any(m in body or m in low for m in _PERMANENT_MARKERS)


def _multipart(fields: dict, image_bytes: bytes, image_name: str) -> tuple[bytes, str]:
    """Encode a multipart/form-data body; return (body, content_type)."""
    boundary = f"----editjudge{uuid.uuid4().hex}"
    crlf = b"\r\n"
    buf = io.BytesIO()
    for key, val in fields.items():
        if val is None:
            continue
        buf.write(f"--{boundary}".encode() + crlf)
        buf.write(f'Content-Disposition: form-data; name="{key}"'.encode() + crlf + crlf)
        buf.write(str(val).encode("utf-8") + crlf)
    buf.write(f"--{boundary}".encode() + crlf)
    buf.write(
        f'Content-Disposition: form-data; name="image"; filename="{image_name}"'.encode()
        + crlf
    )
    buf.write(b"Content-Type: application/octet-stream" + crlf + crlf)
    buf.write(image_bytes + crlf)
    buf.write(f"--{boundary}--".encode() + crlf)
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


class QwenImageEditClient:
    """Edit an image via an OpenAI-compatible ``/v1/images/edits`` endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "qwen-image-edit",
        api_key_env: str = "EDITJUDGE_IMAGE_EDIT_KEY",
        size: Optional[str] = None,
        response_format: Optional[str] = "b64_json",
        timeout: int = 120,
        max_retries: int = 4,
        cache_dir: Optional[Path] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.size = size
        # Ask for the image inline; some relays otherwise return a url whose
        # host Cloudflare-blocks a plain urllib download (HTTP 403 / error 1010).
        self.response_format = response_format
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _endpoint(self) -> str:
        # base_url may or may not already include /v1.
        if self.base_url.endswith("/images/edits"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/images/edits"
        return f"{self.base_url}/v1/images/edits"

    def _cache_key(self, prompt: str, image_bytes: bytes) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(b"\x00" + prompt.encode("utf-8"))
        h.update(b"\x00" + (self.size or "").encode())
        h.update(b"\x00")
        h.update(image_bytes)
        return h.hexdigest()

    def edit(self, image_path: Path, prompt: str) -> Image.Image:
        """Return the edited image. Uses the disk cache when available."""
        image_bytes = Path(image_path).read_bytes()
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{self._cache_key(prompt, image_bytes)}.png"
            if cache_path.exists():
                with Image.open(cache_path) as im:
                    return im.convert("RGB")

        out = self._request(prompt, image_bytes, Path(image_path).name)
        if cache_path is not None:
            out.save(cache_path)
        return out

    # ------------------------------------------------------------------ #
    def _request(self, prompt: str, image_bytes: bytes, image_name: str) -> Image.Image:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ImageEditError(f"env var {self.api_key_env} is empty; cannot call the edit API")
        fields = {
            "model": self.model,
            "prompt": prompt,
            "size": self.size,
            "response_format": self.response_format,
        }
        body, content_type = _multipart(fields, image_bytes, image_name)

        last_err = ""
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self._endpoint(), data=body, method="POST")
            req.add_header("Authorization", f"Bearer {key}")
            req.add_header("Content-Type", content_type)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return self._decode(payload)
            except urllib.error.HTTPError as err:
                last_err = _read_body(err)
                if _is_permanent(last_err) or err.code not in _RETRYABLE_STATUS:
                    raise ImageEditError(f"HTTP {err.code}: {last_err[:300]}") from err
            except _CONNECTION_ERRORS as err:
                # Connection-level failures are transient and MUST be retried rather than
                # escaping. Measured 2026-07-30: a bare `http.client.RemoteDisconnected`
                # ("Remote end closed connection without response") propagated out of this
                # client, sailed past the runner's `except ImageEditError`, and killed a
                # 164-render batch at render 61 — a whole-batch abort from one reset socket,
                # which is exactly what the resumable/fail-soft convention forbids.
                # `RemoteDisconnected` subclasses ConnectionResetError but NOT URLError, so
                # the old two-name tuple could not see it.
                last_err = f"{type(err).__name__}: {err}"
            time.sleep(min(2 ** attempt, 20))
        raise ImageEditError(f"edit failed after {self.max_retries} retries: {last_err[:300]}")

    def _decode(self, payload: dict) -> Image.Image:
        data = (payload.get("data") or [])
        if not data:
            raise ImageEditError(f"no image in response: {str(payload)[:300]}")
        item = data[0]
        if item.get("b64_json"):
            import base64

            raw = base64.b64decode(item["b64_json"])
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if item.get("url"):
            # A browser-like UA avoids Cloudflare bot blocks (HTTP 403 / 1010).
            req = urllib.request.Request(
                item["url"], headers={"User-Agent": "Mozilla/5.0 (editjudge)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        raise ImageEditError(f"response item has neither b64_json nor url: {str(item)[:200]}")


def _read_body(err: urllib.error.HTTPError) -> str:
    try:
        return err.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return str(err)
