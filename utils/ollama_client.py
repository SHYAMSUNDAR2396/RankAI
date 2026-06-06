"""LLM wrapper that funnels every ``ollama.chat`` call through one seam.

``OllamaClient`` is the single place in the system that talks to the locally
hosted Ollama server (Requirement 1.1). Centralizing the call here gives the
rest of the pipeline one consumer-agnostic surface for:

* retry with exponential backoff on transient failures (Requirement 13.7, 13.8),
* DEBUG-level request/response logging (Requirement 13.6), and
* a single swap point for an alternative LLM backend such as the documented
  Groq ``llama-3.1-70b`` free tier (Requirement 1.11).

No other module calls ``ollama.chat`` directly; they depend on this wrapper.

Testability note: the call path goes through the module-level ``ollama.chat``
function on purpose so tests can mock it with
``unittest.mock.patch("ollama.chat")`` (Requirement 12.5). ``host`` is recorded
on the instance for the alternative-backend seam and for logging, but the
module-level ``ollama.chat`` call itself targets the SDK's default host; routing
to a non-default host is a concern of the backend-swap implementation and must
not break the ``patch("ollama.chat")`` seam.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import ollama

import config

logger = logging.getLogger(__name__)

OLLAMA_MODEL_DEFAULT = "llama3.2:3b"  # fallback only

#: Total number of chat attempts: the initial call plus two retries
#: (Requirement 13.7, 13.8).
_MAX_ATTEMPTS = 3

#: Matches a single markdown code fence wrapping the whole string, with an
#: optional ``json`` (or other) info string and surrounding whitespace. The
#: inner payload is captured in group 1.
_CODE_FENCE_RE = re.compile(
    r"\A\s*```[^\n`]*\n?(.*?)\n?\s*```\s*\Z",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding markdown code fence from ``text``.

    Handles ```` ```json ... ``` ```` and bare ```` ``` ... ``` ```` wrappers
    along with any leading/trailing whitespace. When no fence is present the
    input is returned with surrounding whitespace removed.

    Args:
        text: Raw content that may be wrapped in a markdown code fence.

    Returns:
        The inner content with the code fence and surrounding whitespace
        removed, or the whitespace-stripped input when no fence is present.
    """
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


class OllamaCallError(Exception):
    """Raised when an Ollama chat call fails after all retry attempts.

    Signals that the initial attempt and both retries were exhausted without a
    usable response (Requirement 13.8).
    """


class OllamaClient:
    """Wrapper around ``ollama.chat`` with retry, backoff, and DEBUG logging.

    This is the single seam through which all LLM inference flows, which keeps
    retry/backoff and logging logic in one place and provides one swap point for
    an alternative backend (Requirements 1.1, 1.11, 13.6).
    """

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        """Initialize the client.

        Args:
            host: Base URL of the local Ollama server. Falls back to
                ``config.OLLAMA_HOST`` when not provided.
            model: Name of the local model to use for inference. Falls back to
                ``OLLAMA_MODEL_DEFAULT`` when not provided.
        """
        self.host: str = host if host is not None else config.OLLAMA_HOST
        self.model: str = model if model is not None else OLLAMA_MODEL_DEFAULT

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Call ``ollama.chat`` with retry/backoff and return the content string.

        Retries on any raised exception or a falsy/empty response up to two
        additional times (three attempts total). Before each retry it sleeps
        with exponential backoff via ``time.sleep`` starting at 1 second and
        doubling (1s before the first retry, 2s before the second)
        (Requirement 13.7). Request and response detail are emitted at DEBUG
        level on each call (Requirement 13.6).

        Args:
            messages: Chat messages in the Ollama format, e.g.
                ``[{"role": "user", "content": "..."}]``.
            temperature: Sampling temperature. When ``None`` it defaults to
                ``config.SCORING_TEMPERATURE``.
            max_tokens: Maximum tokens to generate. When ``None`` the
                ``num_predict`` option is omitted and the model default applies.
            model: Optional per-call model override.

        Returns:
            The response content string from ``response["message"]["content"]``.

        Raises:
            OllamaCallError: When the initial attempt and both retries fail
                (Requirement 13.8).
        """
        if temperature is None:
            temperature = config.SCORING_TEMPERATURE

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        resolved_model = model or self.model

        logger.debug(
            "ollama.chat request: model=%s host=%s options=%s messages=%s",
            resolved_model,
            self.host,
            options,
            messages,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = ollama.chat(
                    model=resolved_model,
                    messages=messages,
                    options=options,
                )
                content = self._extract_content(response)
                if not content:
                    raise OllamaCallError("Ollama returned an empty response")
                logger.debug(
                    "ollama.chat response (attempt %d/%d): %r",
                    attempt,
                    _MAX_ATTEMPTS,
                    content,
                )
                return content
            except Exception as exc:  # noqa: BLE001 - retry on any failure
                last_error = exc
                logger.debug(
                    "ollama.chat attempt %d/%d failed: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                if attempt < _MAX_ATTEMPTS:
                    backoff = 2 ** (attempt - 1)
                    logger.debug(
                        "backing off %ds before ollama.chat retry", backoff
                    )
                    time.sleep(backoff)

        raise OllamaCallError(
            f"ollama.chat failed after {_MAX_ATTEMPTS} attempts"
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
        """Call :meth:`chat` then parse the response content as JSON.

        The raw content is stripped of any surrounding markdown code fences
        (```` ```json ... ``` ```` or bare ```` ``` ... ``` ````) before being
        passed to :func:`json.loads`. On a JSON parse failure *only*, a warning
        is logged and the caller-supplied ``fallback`` is returned so the
        pipeline can continue without raising (Requirement 13.9). Any other
        exception — including :class:`OllamaCallError` raised by :meth:`chat`
        when all retries are exhausted — propagates unchanged rather than being
        substituted with the fallback (Requirement 13.10).

        Args:
            messages: Chat messages in the Ollama format, e.g.
                ``[{"role": "user", "content": "..."}]``.
            fallback: Value returned when, and only when, the response content
                cannot be parsed as JSON.
            temperature: Sampling temperature forwarded to :meth:`chat`. When
                ``None`` it defaults to ``config.SCORING_TEMPERATURE``.
            max_tokens: Maximum tokens to generate, forwarded to :meth:`chat`.
                When ``None`` the model default applies.
            model: Optional per-call model override.

        Returns:
            The parsed JSON value (a ``dict`` or ``list``) on success, or the
            caller-supplied ``fallback`` when JSON parsing fails.

        Raises:
            OllamaCallError: When the underlying :meth:`chat` call fails after
                all retry attempts (Requirement 13.10).
            Exception: Any non-JSON-parse error raised while obtaining the
                response propagates to the caller (Requirement 13.10).
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
                "Failed to parse JSON from Ollama response; returning fallback "
                "(error=%s content=%r)",
                exc,
                content,
            )
            return fallback

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract the message content from an Ollama chat response.

        The primary path is ``response["message"]["content"]``. Attribute-style
        access (``response.message.content``) is supported defensively so the
        wrapper tolerates both dict-shaped and object-shaped SDK responses.

        Args:
            response: The value returned by ``ollama.chat``.

        Returns:
            The extracted content string, or an empty string when no content is
            present (which the caller treats as a failed attempt).
        """
        message: Any = None
        if isinstance(response, dict):
            message = response.get("message")
        else:
            message = getattr(response, "message", None)

        if message is None:
            return ""

        content: Any
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if not content:
            return ""
        return str(content)
