"""Shared retry classification and backoff helpers for provider API calls."""

from __future__ import annotations

import random
from typing import Iterator

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 405, 422}
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "InternalServerError",
    "RateLimitError",
    "ReadTimeout",
    "RemoteProtocolError",
    "ServiceUnavailableError",
    "TimeoutError",
}
NON_RETRYABLE_ERROR_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "ForbiddenError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnauthorizedError",
    "UnprocessableEntityError",
    "ValidationError",
}
RATE_LIMIT_MESSAGE_TOKENS = (
    "rate limit",
    "too many requests",
    "resource exhausted",
    "quota exceeded",
    "retry after",
)
RETRYABLE_MESSAGE_TOKENS = (
    "connection reset",
    "connection refused",
    "connection aborted",
    "internal server error",
    "overloaded",
    "server error",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "try again",
)
NON_RETRYABLE_MESSAGE_TOKENS = (
    "authentication",
    "bad request",
    "context length",
    "forbidden",
    "invalid api key",
    "permission denied",
    "unauthorized",
    "unsupported",
)


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _extract_status_code(exc: BaseException) -> int | None:
    for current in iter_exception_chain(exc):
        for attr in ("status_code", "status"):
            value = getattr(current, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(current, "response", None)
        if response is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def extract_retry_after_s(exc: BaseException) -> float | None:
    for current in iter_exception_chain(exc):
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None) or getattr(current, "headers", None)
        if not headers:
            continue
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            continue
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            continue
    return None


def _error_message(exc: BaseException) -> str:
    return " | ".join(str(current) for current in iter_exception_chain(exc)).lower()


def classify_retryable_exception(exc: BaseException) -> tuple[bool, str, float | None]:
    """Return (retryable, reason, retry_after_s) for a provider API exception."""
    retry_after_s = extract_retry_after_s(exc)
    status_code = _extract_status_code(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        reason = "rate_limited" if status_code == 429 else f"http_{status_code}"
        return True, reason, retry_after_s
    if status_code in NON_RETRYABLE_STATUS_CODES:
        return False, f"http_{status_code}", None

    class_names = {type(current).__name__ for current in iter_exception_chain(exc)}
    if class_names & RETRYABLE_ERROR_NAMES:
        reason = "rate_limited" if "RateLimitError" in class_names else "transient_error"
        return True, reason, retry_after_s
    if class_names & NON_RETRYABLE_ERROR_NAMES:
        return False, "client_error", None

    message = _error_message(exc)
    if any(token in message for token in RATE_LIMIT_MESSAGE_TOKENS):
        return True, "rate_limited", retry_after_s
    if any(token in message for token in NON_RETRYABLE_MESSAGE_TOKENS):
        return False, "client_error", None
    if any(token in message for token in RETRYABLE_MESSAGE_TOKENS):
        return True, "transient_error", retry_after_s
    return False, "unknown_error", None


def compute_retry_delay(
    attempt_index: int,
    *,
    retry_after_s: float | None,
    base_delay_s: float,
    max_delay_s: float,
    jitter_s: float,
) -> float:
    delay = base_delay_s * (2**attempt_index)
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    delay = min(delay, max_delay_s)
    if jitter_s > 0:
        delay += random.uniform(0, jitter_s)
    return max(0.0, delay)
