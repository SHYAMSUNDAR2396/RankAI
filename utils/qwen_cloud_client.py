"""Qwen Cloud (DashScope) LLM client for the Qwen Cloud Hackathon submission.

This module adds Alibaba Cloud's Qwen Cloud as a third LLM backend alongside
Ollama (local) and Groq (cloud). It mirrors :class:`utils.ollama_client.OllamaClient`
so the rest of the pipeline stays backend-agnostic — every existing caller
(``chat_json``/``chat``) continues to work unchanged.

DashScope exposes an OpenAI-compatible ``/compatible-mode/v1`` endpoint, so we
reuse the official ``openai`` Python SDK rather than pulling in a separate
DashScope SDK. Authentication is via ``DASHSCOPE_API_KEY`` (env var or
``config.QWEN_CLOUD_API_KEY``).

Usage:

    LLM_BACKEND=qwen_cloud DASHSCOPE_API_KEY=sk-... python main.py

Testability: the ``openai.OpenAI`` client is constructed lazily and is replaced
by a mock at test time via ``unittest.mock.patch("openai.OpenAI")``. All retry,
backoff, and JSON-fence-stripping logic is identical to the Ollama client so
behaviour parity is preserved across backends.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

try:
    import openai
except ImportError:  # pragma: no cover - SDK is a required runtime dep
    openai = None

import config

logger = logging.getLogger(__name__)

QWEN_DEFAULT_MODEL = "qwen-turbo"

_MAX_ATTEMPTS = 3

_CODE_FENCE_RE = re.compile(
    r"\A\s*```[^\n`]*\n?(.*?)\n?\s*```\s*\Z",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding markdown code fence from ``text``."""
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


class QwenCloudCallError(Exception):
    """Raised when a Qwen Cloud chat call fails after all retry attempts."""


class QwenCloudClient:
    """Wrapper around DashScope's OpenAI-compatible chat completions endpoint.

    Mirrors :class:`utils.ollama_client.OllamaClient`:

    * retry with exponential backoff on any transient failure (1s, then 2s),
    * DEBUG-level request/response logging,
    * a single swap point for the LLM backend, and
    * a ``chat_json`` helper that strips markdown fences and falls back to a
      caller-supplied default on JSON parse failure only.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the Qwen Cloud client.

        Args:
            api_key: DashScope API key. Falls back to
                ``config.QWEN_CLOUD_API_KEY`` or the ``DASHSCOPE_API_KEY`` env
                var.
            base_url: Endpoint base URL. Falls back to
                ``config.QWEN_CLOUD_BASE_URL``.
            model: Default model name. Falls back to :data:`QWEN_DEFAULT_MODEL`.
        """
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for the Qwen Cloud backend. "
                "Install it with: pip install openai>=1.0.0"
            )

        resolved_key = (
            api_key
            or config.QWEN_CLOUD_API_KEY
            or os.environ.get("DASHSCOPE_API_KEY", "")
        )
        if not resolved_key:
            raise ValueError(
                "DASHSCOPE_API_KEY must be set when LLM_BACKEND is 'qwen_cloud'. "
                "Request a free key at https://dashscope.aliyun.com/."
            )

        self.api_key: str = resolved_key
        self.base_url: str = (
            base_url
            if base_url is not None
            else config.QWEN_CLOUD_BASE_URL
        )
        self.model: str = model if model is not None else QWEN_DEFAULT_MODEL
        self._client = openai.OpenAI(api_key=resolved_key, base_url=self.base_url)
        logger.info(
            "Qwen Cloud client initialized (base_url=%s, default_model=%s)",
            self.base_url,
            self.model,
        )

    def _completion_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        model: str,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        return kwargs

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Call Qwen Cloud chat completion with retry/backoff and return content.

        Args:
            messages: OpenAI-format chat messages.
            temperature: Sampling temperature. Defaults to
                ``config.SCORING_TEMPERATURE`` when ``None``.
            max_tokens: Maximum tokens to generate. ``None`` → provider default.
            model: Optional per-call model override.

        Returns:
            str: The assistant message content.

        Raises:
            QwenCloudCallError: When the initial attempt and both retries fail.
        """
        if temperature is None:
            temperature = config.SCORING_TEMPERATURE

        resolved_model = model or self.model
        kwargs = self._completion_kwargs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=resolved_model,
        )

        logger.debug(
            "qwen_cloud.chat request: model=%s temperature=%s max_tokens=%s messages=%s",
            resolved_model,
            temperature,
            max_tokens,
            messages,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = self._extract_content(response)
                if not content:
                    raise QwenCloudCallError(
                        f"qwen_cloud.chat returned an empty response (attempt {attempt})"
                    )
                logger.debug(
                    "qwen_cloud.chat response (attempt %d/%d): %r",
                    attempt,
                    _MAX_ATTEMPTS,
                    content,
                )
                return content
            except QwenCloudCallError:
                raise
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "qwen_cloud.chat attempt %d/%d failed: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                if attempt < _MAX_ATTEMPTS:
                    backoff = 2 ** (attempt - 1)
                    logger.debug(
                        "backing off %ds before qwen_cloud.chat retry", backoff
                    )
                    time.sleep(backoff)

        raise QwenCloudCallError(
            f"qwen_cloud.chat failed after {_MAX_ATTEMPTS} attempts"
        ) from last_error

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        fallback: dict | list,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> dict | list:
        """Call :meth:`chat` and parse the response content as JSON.

        Behaves identically to :meth:`utils.ollama_client.OllamaClient.chat_json`:
        strip markdown fences, parse JSON, return the supplied ``fallback`` on
        parse failure only. Other exceptions (including retry exhaustion) propagate.

        Args:
            messages: Chat messages.
            fallback: Value to return when JSON parsing fails.
            temperature: Sampling temperature forwarded to :meth:`chat`.
            max_tokens: Maximum tokens forwarded to :meth:`chat`.
            model: Optional per-call model override.

        Returns:
            dict | list: The parsed JSON value, or the caller-supplied fallback
            when JSON parsing fails.
        """
        content = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        stripped = _strip_code_fences(content)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse JSON from Qwen Cloud response; returning fallback "
                "(error=%s content=%r)",
                exc,
                content,
            )
            return fallback

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract message content from an OpenAI chat response.

        Tolerates both object-style (.choices[0].message.content) and
        dict-shaped responses for robustness across SDK versions.

        Args:
            response: OpenAI chat completion response.

        Returns:
            str: The assistant message content, or an empty string when absent.
        """
        choices = getattr(response, "choices", None) or response.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None) or choice.get("message")
        if message is None:
            return ""
        content = getattr(message, "content", None) or message.get("content")
        return str(content or "")
