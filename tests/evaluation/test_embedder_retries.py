import pytest

from retrieval_evaluator.evaluation import embedder as embedder_module
from retrieval_evaluator.evaluation.embedder import _request_with_retries


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(embedder_module.time, "sleep", lambda _s: None)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeBadRequestError(Exception):
    status_code = 400


def test_request_with_retries_recovers_from_rate_limit():
    calls = {"count": 0}

    def request():
        calls["count"] += 1
        if calls["count"] < 3:
            raise FakeRateLimitError("Too many requests")
        return "ok"

    assert _request_with_retries(request, description="test request") == "ok"
    assert calls["count"] == 3


def test_request_with_retries_does_not_retry_client_errors():
    calls = {"count": 0}

    def request():
        calls["count"] += 1
        raise FakeBadRequestError("Bad request")

    with pytest.raises(FakeBadRequestError):
        _request_with_retries(request, description="test request")
    assert calls["count"] == 1


def test_request_with_retries_gives_up_after_max_attempts():
    calls = {"count": 0}

    def request():
        calls["count"] += 1
        raise FakeRateLimitError("Too many requests")

    with pytest.raises(FakeRateLimitError):
        _request_with_retries(request, description="test request")
    assert calls["count"] == embedder_module._API_MAX_ATTEMPTS
