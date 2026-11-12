import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict

import pandas as pd

from retrieval_evaluator.config import get_config
from retrieval_evaluator.utils.retry import classify_retryable_exception, compute_retry_delay

logger = logging.getLogger(__name__)

_llm_cache: dict[tuple[str, float | None], object] = {}

# Hint shown when the LangChain provider package for the configured model is missing.
_PROVIDER_INSTALL_HINTS = {
    "openai": 'pip install "retrieval-evaluator[openai]"',
    "azure_openai": 'pip install "retrieval-evaluator[openai]"',
    "anthropic": 'pip install "retrieval-evaluator[anthropic]"',
}


class MalformedLLMResponseError(ValueError):
    pass


def _make_query_id(row: pd.Series, row_index: object) -> str:
    source_id = row.get("id")
    if source_id is not None and not pd.isna(source_id):
        stable_source = f"id:{source_id}"
    else:
        stable_source = (
            f"row:{row_index}|doc:{row.get('doc_id')}|chunk:{row.get('chunk_id')}|text:{row.get('text')}"
        )
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"retrieval-evaluator-query:{stable_source}"))


def _llm_settings() -> dict[str, Any]:
    llm_cfg = get_config().get("llm", {})
    timeout_s = llm_cfg.get("timeout_s", 60)
    if timeout_s is not None:
        timeout_s = float(timeout_s)
        if timeout_s <= 0:
            raise ValueError("llm.timeout_s must be positive or null")
    max_attempts = int(llm_cfg.get("max_attempts", 5))
    if max_attempts <= 0:
        raise ValueError("llm.max_attempts must be positive")
    retry_base_delay_s = float(llm_cfg.get("retry_base_delay_s", 1.0))
    retry_max_delay_s = float(llm_cfg.get("retry_max_delay_s", 30.0))
    retry_jitter_s = float(llm_cfg.get("retry_jitter_s", 0.5))
    if min(retry_base_delay_s, retry_max_delay_s, retry_jitter_s) < 0:
        raise ValueError("LLM retry delays must be zero or greater")
    return {
        "model": llm_cfg.get("model"),
        "max_attempts": max_attempts,
        "timeout_s": timeout_s,
        "retry_base_delay_s": retry_base_delay_s,
        "retry_max_delay_s": retry_max_delay_s,
        "retry_jitter_s": retry_jitter_s,
        "retry_on_invalid_response": bool(llm_cfg.get("retry_on_invalid_response", True)),
    }


def _get_llm(model: str | None = None, timeout_s: float | None = None):
    try:
        from langchain.chat_models import init_chat_model  # lazy — only needed for build-dataset
    except ImportError as exc:
        raise ImportError(
            "Dataset generation requires the dataset extra plus the configured LLM provider extra. "
            'For example: pip install "retrieval-evaluator[dataset,openai]"'
        ) from exc

    model = model or _llm_settings()["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("llm.model must be a non-empty model identifier")
    cache_key = (model, timeout_s)
    if cache_key not in _llm_cache:
        try:
            kwargs: dict[str, Any] = {"max_retries": 0}
            if timeout_s is not None:
                kwargs["timeout"] = timeout_s
            _llm_cache[cache_key] = init_chat_model(model, **kwargs)
        except ImportError as exc:
            provider = model.split(":", 1)[0] if ":" in model else model
            hint = _PROVIDER_INSTALL_HINTS.get(provider, "install the matching langchain provider package")
            raise ImportError(
                f"The LangChain provider package for llm model '{model}' is not installed. "
                f"Install it with: {hint}"
            ) from exc
    return _llm_cache[cache_key]


def _build_messages(system_prompt: str, prompt: str) -> list:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        raise ImportError(
            "Dataset generation requires the dataset extra. "
            'Install with: pip install "retrieval-evaluator[dataset]"'
        ) from exc
    return [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(content)


def call_llm(
    prompt: str,
    system_prompt: str,
    max_retries: int | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    retry_base_delay_s: float | None = None,
    retry_max_delay_s: float | None = None,
    retry_jitter_s: float | None = None,
) -> str:
    settings = _llm_settings()
    max_attempts = max(1, max_retries if max_retries is not None else settings["max_attempts"])
    timeout_s = settings["timeout_s"] if timeout_s is None else timeout_s
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive or None")
    base_delay_s = settings["retry_base_delay_s"] if retry_base_delay_s is None else retry_base_delay_s
    max_delay_s = settings["retry_max_delay_s"] if retry_max_delay_s is None else retry_max_delay_s
    jitter_s = settings["retry_jitter_s"] if retry_jitter_s is None else retry_jitter_s
    llm = _get_llm(model=model, timeout_s=timeout_s)
    messages = _build_messages(system_prompt, prompt)
    last_error: BaseException | None = None
    for attempt_index in range(max_attempts):
        try:
            return _response_to_text(llm.invoke(messages))
        except Exception as exc:
            last_error = exc
            retryable, reason, retry_after_s = classify_retryable_exception(exc)
            if not retryable or attempt_index == max_attempts - 1:
                break
            delay_s = compute_retry_delay(
                attempt_index,
                retry_after_s=retry_after_s,
                base_delay_s=base_delay_s,
                max_delay_s=max_delay_s,
                jitter_s=jitter_s,
            )
            logger.warning(
                "LLM call failed (%s, attempt %s/%s): %s. Retrying in %.2fs",
                reason,
                attempt_index + 1,
                max_attempts,
                exc,
                delay_s,
            )
            time.sleep(delay_s)
    raise RuntimeError(f"LLM call failed after {max_attempts} attempts") from last_error


def parse_llm_json_response(text: str) -> Dict:
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    json_string = match.group(1) if match else text.strip()
    return json.loads(json_string)


def _parse_generated_query(output: str) -> tuple[str, str | None]:
    try:
        output_dict = parse_llm_json_response(output)
    except json.JSONDecodeError as exc:
        raise MalformedLLMResponseError("LLM response was not valid JSON") from exc
    if not isinstance(output_dict, dict):
        raise MalformedLLMResponseError("LLM response JSON must be an object")

    question = output_dict.get("question")
    if not isinstance(question, str) or not question.strip():
        raise MalformedLLMResponseError("LLM response did not include a non-empty 'question'")

    language = output_dict.get("language")
    if language is not None:
        language = str(language).strip() or None

    return question.strip(), language


def generate_queries_for_chunks(
    df: pd.DataFrame,
    generate_fn: Callable[[str, str], str] | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    sleep_s: float = 1.0,
    chunk_limit: int = 200,
    llm_model: str | None = None,
    max_attempts: int | None = None,
    timeout_s: float | None = None,
    retry_base_delay_s: float | None = None,
    retry_max_delay_s: float | None = None,
    retry_jitter_s: float | None = None,
    retry_on_invalid_response: bool | None = None,
) -> tuple[list[str | None], list[str | None], list[str | None]]:
    """
    For each chunk, generate a query and language using the LLM.
    Returns (queries, qids, langs) lists aligned with the input rows; entries are
    None for rows where generation failed or was skipped by chunk_limit.
    """
    cfg = get_config()
    llm_settings = _llm_settings()
    if chunk_limit <= 0:
        raise ValueError("chunk_limit must be positive")
    if sleep_s < 0:
        raise ValueError("sleep_s must be zero or greater")
    if system_prompt is None:
        system_prompt = cfg["retrieval_evaluation"]["dataset"]["query_generation_system_prompt"]
    if user_prompt_template is None:
        user_prompt_template = cfg["retrieval_evaluation"]["dataset"]["user_prompt_template"]
    max_attempts = llm_settings["max_attempts"] if max_attempts is None else max(1, max_attempts)
    timeout_s = llm_settings["timeout_s"] if timeout_s is None else timeout_s
    base_delay_s = llm_settings["retry_base_delay_s"] if retry_base_delay_s is None else retry_base_delay_s
    max_delay_s = llm_settings["retry_max_delay_s"] if retry_max_delay_s is None else retry_max_delay_s
    jitter_s = llm_settings["retry_jitter_s"] if retry_jitter_s is None else retry_jitter_s
    retry_on_invalid_response = (
        llm_settings["retry_on_invalid_response"]
        if retry_on_invalid_response is None
        else retry_on_invalid_response
    )
    if generate_fn is None:
        # Build the LLM client once up front so a missing provider package or
        # missing API key aborts the run immediately with the real cause,
        # instead of surfacing as one buried warning per chunk.
        _get_llm(model=llm_model, timeout_s=timeout_s)

        def _default_generate_fn(prompt, current_system_prompt):
            # max_retries=1 here is intentional: the outer loop in generate_queries_for_chunks
            # handles per-chunk retries up to max_attempts. The inner call_llm does not
            # retry so the outer loop can classify and count each failure independently.
            return call_llm(
                prompt,
                current_system_prompt,
                max_retries=1,
                model=llm_model,
                timeout_s=timeout_s,
                retry_base_delay_s=base_delay_s,
                retry_max_delay_s=max_delay_s,
                retry_jitter_s=jitter_s,
            )
        generate_fn = _default_generate_fn

    queries, qids, langs = [], [], []
    n_generated = 0
    limit_logged = False

    for row_index, row in df.iterrows():
        if "query" in row and not pd.isna(row["query"]):
            existing_query = row["query"]
            if not isinstance(existing_query, str) or not existing_query.strip():
                raise ValueError("Existing query values must be non-empty strings")
            existing_qid = row.get("qid")
            if existing_qid is None or pd.isna(existing_qid) or not str(existing_qid).strip():
                existing_qid = _make_query_id(row, row_index)
            queries.append(existing_query.strip())
            qids.append(str(existing_qid))
            existing_language = row.get("language")
            langs.append(None if existing_language is None or pd.isna(existing_language) else str(existing_language))
            continue

        if n_generated >= chunk_limit:
            if not limit_logged:
                logger.info(f"Reached chunk_limit={chunk_limit}; remaining rows will be skipped")
                limit_logged = True
            queries.append(None)
            qids.append(None)
            langs.append(None)
            continue

        prompt = user_prompt_template.format(passage=row.text)
        question, qid, lang = None, None, None
        row_id = row.get("id", None)
        for attempt_index in range(max_attempts):
            try:
                output = generate_fn(prompt, system_prompt)
                question, lang = _parse_generated_query(output)
                qid = _make_query_id(row, row_index)
                n_generated += 1
                break
            except Exception as exc:
                # A missing provider package can never succeed on retry or on the
                # next chunk — abort the whole run with the install hint intact.
                if isinstance(exc, ImportError):
                    raise
                if isinstance(exc.__cause__, ImportError):
                    raise exc.__cause__ from exc

                malformed_response = isinstance(exc, MalformedLLMResponseError)
                if malformed_response and not retry_on_invalid_response:
                    retryable = False
                    reason = "invalid_response"
                    retry_after_s = None
                elif malformed_response:
                    retryable = True
                    reason = "invalid_response"
                    retry_after_s = None
                else:
                    retryable, reason, retry_after_s = classify_retryable_exception(exc)

                if not retryable or attempt_index == max_attempts - 1:
                    logger.error(
                        "LLM failure on %s (%s, attempt %s/%s): %s",
                        row_id,
                        reason,
                        attempt_index + 1,
                        max_attempts,
                        exc,
                    )
                    break

                delay_s = compute_retry_delay(
                    attempt_index,
                    retry_after_s=retry_after_s,
                    base_delay_s=base_delay_s,
                    max_delay_s=max_delay_s,
                    jitter_s=jitter_s,
                )
                logger.warning(
                    "Retrying query generation for %s after %s (attempt %s/%s) in %.2fs",
                    row_id,
                    reason,
                    attempt_index + 1,
                    max_attempts,
                    delay_s,
                )
                time.sleep(delay_s)

        queries.append(question)
        qids.append(qid)
        langs.append(lang)

        if sleep_s:
            time.sleep(sleep_s)

    return queries, qids, langs
