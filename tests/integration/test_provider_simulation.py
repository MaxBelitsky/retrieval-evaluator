"""Real SDK serialization, parsing and error handling, with no network or API keys."""

import json

import httpx
import pandas as pd
import pytest

from retrieval_evaluator import evaluate
from retrieval_evaluator.dataset import query_generation
from retrieval_evaluator.dataset.beir import write_beir_raw
from retrieval_evaluator.evaluation.embedder import Embedder
from retrieval_evaluator.evaluation.embedders_evaluation import EvaluationRunError


def embedding_response(provider, body):
    # The first passage is relevant; query vectors should rank it first.
    texts = body["input"] if provider == "openai" else body["texts"]
    vectors = [[0.0, 1.0] if text == "Distractor" else [1.0, 0.0] for text in texts]
    if provider == "openai":
        return {
            "object": "list", "model": body["model"],
            "data": [{"object": "embedding", "index": i, "embedding": vector} for i, vector in enumerate(vectors)],
            "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
        }
    return {"id": "offline", "texts": texts, "embeddings": {"float": vectors}, "response_type": "embeddings_by_type"}


@pytest.mark.parametrize("provider, model", [("openai", "text-embedding-3-small"), ("cohere", "embed-multilingual-v3.0")])
def test_sdk_embeddings_reach_rankings_and_saved_results(simulated_http, tmp_path, provider, model):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert request.headers["authorization"] == "Bearer offline-test-key"
        assert body["model"] == model
        return httpx.Response(200, json=embedding_response(provider, body))

    simulated_http(respond)
    write_beir_raw(
        [{"_id": "relevant", "text": "Answer"}, {"_id": "decoy", "text": "Distractor"}],
        [{"_id": "q1", "text": "Question?"}],
        {"q1": {"relevant": 1}},
        tmp_path / "data" / "test",
    )
    results = evaluate(
        dataset_dir=tmp_path / "data", collections=["test"],
        embedder_configs={"candidate": {"provider": provider, "model": model}},
        batch_size=1, artifacts_dir=tmp_path / "runs", index_backend="numpy",
    )
    assert results.iloc[0]["R@1"] == results.iloc[0]["MRR@10"] == 1.0
    run_dir = tmp_path / "runs" / results.iloc[0]["run_idx"]
    assert json.loads((run_dir / "rankings.json").read_text()) == {"q1": ["relevant", "decoy"]}
    assert len(requests) == 3  # Two passage batches and one query batch.
    if provider == "cohere":
        assert [body["input_type"] for _, body in requests] == ["search_document", "search_document", "search_query"]
        assert all(path == "/v2/embed" for path, _ in requests)
    else:
        assert all(path == "/v1/embeddings" for path, _ in requests)


@pytest.mark.parametrize("provider", ["openai", "cohere"])
@pytest.mark.parametrize("failure", [429, 401, "timeout"])
def test_real_sdk_errors_retry_only_transient_failures(simulated_http, monkeypatch, provider, failure):
    calls = []
    delays = []
    monkeypatch.setattr("retrieval_evaluator.evaluation.embedder.time.sleep", delays.append)

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("simulated timeout", request=request)
            return httpx.Response(
                failure, headers={"retry-after": "2"},
                json={"message": "simulated failure", "error": {"message": "simulated failure", "type": "api_error"}},
            )
        return httpx.Response(200, json=embedding_response(provider, json.loads(request.content)))

    simulated_http(respond)
    embedder = Embedder({"provider": provider, "model": "offline-model"})
    if failure == 401:
        with pytest.raises(Exception) as error:
            embedder.encode(["Question?"], batch_size=1)
        assert getattr(error.value, "status_code", None) == 401
        assert len(calls) == 1 and not delays
    else:
        assert embedder.encode(["Question?"], batch_size=1).tolist() == [[1.0, 0.0]]
        assert len(calls) == 2 and len(delays) == 1
        if failure == 429:
            assert delays[0] >= 2


@pytest.mark.parametrize("provider, model", [("openai", "gpt-4o"), ("anthropic", "claude-sonnet-4-6")])
@pytest.mark.parametrize("failure", ["invalid_json", 429])
def test_langchain_sdk_query_generation_recovers(simulated_http, provider, model, failure):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == model
        assert "Passage about refunds" in body["messages"][-1]["content"]
        if len(requests) == 1 and failure == 429:
            return httpx.Response(429, json={
                "type": "error", "error": {"type": "rate_limit_error", "message": "simulated rate limit"},
            })
        content = "invalid JSON" if len(requests) == 1 else '{"question":"How do refunds work?","language":"en"}'
        if provider == "openai":
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            })
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "offline-test-key"
        return httpx.Response(200, json={
            "id": "offline", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": content}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        })

    simulated_http(respond)
    query_generation._llm_cache.clear()
    try:
        questions, qids, languages = query_generation.generate_queries_for_chunks(
            pd.DataFrame([{"id": "c1", "text": "Passage about refunds"}]),
            llm_model=f"{provider}:{model}", sleep_s=0, max_attempts=2,
            retry_base_delay_s=0, retry_max_delay_s=0, retry_jitter_s=0,
        )
    finally:
        query_generation._llm_cache.clear()
    assert questions == ["How do refunds work?"]
    assert len(qids) == 1 and qids[0]
    assert languages == ["en"]
    assert len(requests) == 2


@pytest.mark.parametrize("provider", ["openai", "cohere"])
@pytest.mark.parametrize("failure", [503, "wrong_vector_count"])
def test_failed_provider_runs_do_not_publish_scores(simulated_http, monkeypatch, tmp_path, provider, failure):
    calls = []
    monkeypatch.setattr("retrieval_evaluator.evaluation.embedder.time.sleep", lambda _: None)

    def respond(request):
        calls.append(request)
        if failure == 503:
            return httpx.Response(503, json={
                "message": "simulated outage", "error": {"type": "api_error", "message": "simulated outage"},
            })
        payload = embedding_response(provider, json.loads(request.content))
        if provider == "openai":
            payload["data"] = []
        else:
            payload["embeddings"]["float"] = []
        return httpx.Response(200, json=payload)

    simulated_http(respond)
    write_beir_raw(
        [{"_id": "c1", "text": "Answer"}], [{"_id": "q1", "text": "Question?"}],
        {"q1": {"c1": 1}}, tmp_path / "data" / "test",
    )
    with pytest.raises(EvaluationRunError):
        evaluate(
            dataset_dir=tmp_path / "data", collections=["test"],
            embedder_configs={"candidate": {"provider": provider, "model": "offline-model"}},
            artifacts_dir=tmp_path / "runs", index_backend="numpy",
        )
    assert len(calls) == (5 if failure == 503 else 1)
    assert not list((tmp_path / "runs").rglob("*"))
