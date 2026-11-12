import functools
import gc
import hashlib
import logging
import math
import numbers
import os
import pickle
import time
from typing import Any, Callable, Dict, Literal, Sequence, Union

import numpy as np

from retrieval_evaluator.utils.constants import EmbeddingTextType
from retrieval_evaluator.utils.retry import classify_retryable_exception, compute_retry_delay

logger = logging.getLogger(__name__)

# Retry policy for embedding-API batch requests (OpenAI, Cohere).
_API_MAX_ATTEMPTS = 5
_API_RETRY_BASE_DELAY_S = 1.0
_API_RETRY_MAX_DELAY_S = 30.0
_API_RETRY_JITTER_S = 0.5


def _request_with_retries(request_fn: Callable[[], Any], description: str) -> Any:
    """Run one API request, retrying transient failures with exponential backoff."""
    last_error: BaseException | None = None
    for attempt_index in range(_API_MAX_ATTEMPTS):
        try:
            return request_fn()
        except Exception as exc:
            last_error = exc
            retryable, reason, retry_after_s = classify_retryable_exception(exc)
            if not retryable or attempt_index == _API_MAX_ATTEMPTS - 1:
                raise
            delay_s = compute_retry_delay(
                attempt_index,
                retry_after_s=retry_after_s,
                base_delay_s=_API_RETRY_BASE_DELAY_S,
                max_delay_s=_API_RETRY_MAX_DELAY_S,
                jitter_s=_API_RETRY_JITTER_S,
            )
            logger.warning(
                "%s failed (%s, attempt %s/%s): %s. Retrying in %.2fs",
                description,
                reason,
                attempt_index + 1,
                _API_MAX_ATTEMPTS,
                exc,
                delay_s,
            )
            time.sleep(delay_s)
    raise RuntimeError(f"{description} failed after {_API_MAX_ATTEMPTS} attempts") from last_error

class Embedder:
    """
    Wraps HuggingFace or other embedding model, for BEIR evaluation.
    """
    def __init__(self, cfg: Dict[str, Any], device="cpu", max_batch_size: int = 2048):
        self.cfg = dict(cfg)
        self.max_batch_size = int(max_batch_size)
        requested_max_tokens = cfg.get("max_tokens")
        if requested_max_tokens is not None:
            requested_max_tokens = int(requested_max_tokens)
            if requested_max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")

        self._embeddings_cache = {}

        if cfg["provider"] == "hf":
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                from transformers.utils import logging as transformers_logging
            except ImportError as exc:
                raise ImportError(
                    "HuggingFace embeddings require the hf extra. "
                    'Install with: pip install "retrieval-evaluator[hf]"'
                ) from exc

            transformers_logging.disable_progress_bar()
            _PRECISIONS = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            device_name = device if isinstance(device, str) else getattr(device, "type", str(device))
            self.device = torch.device(device_name)
            precision_name = str(cfg.get("precision", "float32")).removeprefix("torch.")
            if precision_name not in _PRECISIONS:
                supported = ", ".join(sorted(_PRECISIONS))
                raise ValueError(f"Unsupported precision {cfg.get('precision')!r}. Choose one of: {supported}")
            self.precision = precision_name
            self.model = SentenceTransformer(
                cfg["model"],
                model_kwargs=dict(torch_dtype=_PRECISIONS[precision_name]),
                device=self.device,
            )
            model_limit = getattr(self.model, "max_seq_length", None)
            supported_max_tokens = (
                int(model_limit)
                if isinstance(model_limit, numbers.Integral) and int(model_limit) > 0
                else None
            )
            if requested_max_tokens is None:
                self.max_tokens = supported_max_tokens or 512
            else:
                if supported_max_tokens is not None and requested_max_tokens > supported_max_tokens:
                    raise ValueError(
                        f"Configured max_tokens={requested_max_tokens} exceeds the model's "
                        f"supported context length ({supported_max_tokens})"
                    )
                self.max_tokens = requested_max_tokens
                self.model.max_seq_length = requested_max_tokens
        elif cfg["provider"] == "openai":
            self.device = device
            self.max_tokens = requested_max_tokens or 512
            timeout_s = float(cfg.get("timeout_s", 60.0))
            if timeout_s <= 0:
                raise ValueError("timeout_s must be positive")
            try:
                import openai as _openai
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires the openai package. "
                    'Install with: pip install "retrieval-evaluator[openai]"'
                )
            self._openai_client = _openai.OpenAI(timeout=timeout_s, max_retries=0)
        elif cfg["provider"] == "cohere":
            self.device = device
            self.max_tokens = requested_max_tokens or 512
            timeout_s = float(cfg.get("timeout_s", 60.0))
            if timeout_s <= 0:
                raise ValueError("timeout_s must be positive")
            try:
                import cohere as _cohere
            except ImportError:
                raise ImportError(
                    "Cohere provider requires the cohere package. "
                    'Install with: pip install "retrieval-evaluator[cohere]"'
                )
            self._cohere_client = _cohere.ClientV2(timeout=timeout_s)
        else:
            raise ValueError(f"Provider '{cfg['provider']}' is not supported")

    @property
    def identity_config(self) -> dict[str, Any]:
        """Configuration values that materially identify this embedder run."""
        identity = {**self.cfg, "max_tokens": self.max_tokens}
        if self.cfg["provider"] == "hf":
            identity["precision"] = self.precision
        return identity

    def _embedding_cache_key(
        self,
        texts: Sequence[str],
        *,
        text_type: str,
    ) -> str:
        """
        Create a cache key that reflects both the raw texts and prompting mode.
        """
        prompt = self.cfg.get(f"{text_type}_prompt")
        prompt_name = self.cfg.get("prompt_name")
        cache_payload = (tuple(texts), text_type, prompt, prompt_name)
        return hashlib.sha256(pickle.dumps(cache_payload)).hexdigest()

    def clear_cache(self):
        self._embeddings_cache.clear()

    def _resolve_auto_batch_size(self, texts: Sequence[str]) -> int:
        if self.cfg["provider"] == "hf":
            if self.device.type in ["cuda", "mps"]:
                return self.determine_batch_size(texts)
            return 1
        if self.cfg["provider"] == "openai":
            return 512
        if self.cfg["provider"] == "cohere":
            return 96
        return 1

    def _encode_hf(self, texts: Sequence[str], batch_size: int, text_type: str) -> np.ndarray:
        prompt = self.cfg.get(f"{text_type}_prompt")
        prompt_name = self.cfg.get("prompt_name")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            prompt=prompt,
            prompt_name=prompt_name,
            normalize_embeddings=True,
        )

    def _encode_openai(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            response = _request_with_retries(
                lambda: self._openai_client.embeddings.create(
                    input=batch,
                    model=self.cfg["model"],
                ),
                description=f"OpenAI embeddings request ({self.cfg['model']})",
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return np.array(all_embeddings, dtype=np.float32)

    def _encode_cohere(self, texts: Sequence[str], batch_size: int, text_type: str) -> np.ndarray:
        input_type = (
            "search_document" if text_type == EmbeddingTextType.passage else "search_query"
        )
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            response = _request_with_retries(
                lambda: self._cohere_client.embed(
                    texts=batch,
                    model=self.cfg["model"],
                    input_type=input_type,
                    embedding_types=["float"],
                    request_options={"max_retries": 0},
                ),
                description=f"Cohere embeddings request ({self.cfg['model']})",
            )
            all_embeddings.extend(response.embeddings.float_)
        return np.array(all_embeddings, dtype=np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int | Literal["auto"],
        text_type: str = EmbeddingTextType.query,
    ) -> np.ndarray:
        """
        Returns float32 numpy array (n x dim) of embeddings.
        """
        if text_type not in EmbeddingTextType.values():
            raise ValueError(
                f"Unknown text_type {text_type!r}. Expected one of: {list(EmbeddingTextType.values())}"
            )
        if not texts:
            raise ValueError("texts must contain at least one item")
        if any(not isinstance(item, str) or not item.strip() for item in texts):
            raise ValueError("texts must contain only non-empty strings")
        if batch_size != "auto":
            if not isinstance(batch_size, numbers.Integral) or int(batch_size) <= 0:
                raise ValueError("batch_size must be a positive integer or 'auto'")
            batch_size = int(batch_size)

        key = self._embedding_cache_key(texts, text_type=text_type)
        if key in self._embeddings_cache:
            logger.info("Using cached embeddings")
            return self._embeddings_cache[key]

        if batch_size == "auto":
            batch_size = self._resolve_auto_batch_size(texts)

        # logger.info(f"Setting batch size to {batch_size}")

        if self.cfg["provider"] == "hf":
            embeddings = self._encode_hf(texts, batch_size, text_type)
        elif self.cfg["provider"] == "openai":
            embeddings = self._encode_openai(texts, batch_size)
        elif self.cfg["provider"] == "cohere":
            embeddings = self._encode_cohere(texts, batch_size, text_type)
        else:
            raise ValueError(f"Provider '{self.cfg['provider']}' is not supported")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
            raise ValueError(
                f"Embedding provider returned shape {embeddings.shape}; expected ({len(texts)}, dimensions)"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding provider returned non-finite values")
        self._embeddings_cache[key] = embeddings
        return embeddings

    @property
    def supports_tokenization(self) -> bool:
        return self.cfg["provider"] == "hf"

    def tokenize(self, texts):
        if not self.supports_tokenization:
            raise NotImplementedError(
                f"tokenize() is not supported for provider '{self.cfg['provider']}'"
            )
        if isinstance(texts, str):
            texts = [texts]
        return self.model.tokenize(texts)['input_ids']

    def determine_batch_size(self, sample_texts: Sequence[str]) -> int:
        if not self.supports_tokenization:
            raise NotImplementedError(
                f"determine_batch_size() is not supported for provider '{self.cfg['provider']}'"
            )
        import torch
        from accelerate import find_executable_batch_size
        texts = sample_texts
        # Determine the max batch size
        closest_batch_size = get_closest_batch_size(len(texts))
        max_batch_size = min(closest_batch_size, self.max_batch_size)

        # Determine the max sequence length
        max_length = self.tokenize(texts).shape[-1]

        # Set the correct decorator function based on the device
        find_executable_batch_size_wrapper = (
            find_executable_batch_size_mps
            if self.device.type == "mps"
            else find_executable_batch_size
        )

        @find_executable_batch_size_wrapper(starting_batch_size=max_batch_size)
        def _forward_batch(batch_size):
            logger.info(f"Trying batch_size {batch_size}")
            test_batch = torch.ones((batch_size, max_length), device=self.device).long()
            # SentenceTransformer.forward takes the features as one positional dict.
            features = {"input_ids": test_batch, "attention_mask": test_batch}

            for _ in range(3):
                self.model.forward(features)

            cleanup_memory(self.device)
            return batch_size

        try:
            batch_size = _forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        # Clear the cache
        cleanup_memory(self.device)

        return batch_size

    def __repr__(self):
        config_str = "\n".join([f"\t{k}={v}" for k, v in self.cfg.items()])
        return f"""Embedder(\n{config_str}\n)"""

def find_executable_batch_size_mps(starting_batch_size=128, auto_increment=False, max_batch_size=None):
    """
    Drop-in replacement for accelerate.find_executable_batch_size that works with MPS.

    Args:
        starting_batch_size (int): The initial batch size to try.
        auto_increment (bool): If True, increase batch size up to max_batch_size; else, decrease on OOM.
        max_batch_size (Optional[int]): Only used with auto_increment. Upper limit for batch size.

    Returns:
        Callable: A decorator for your training or validation function.

    The decorated function must accept a `batch_size` argument.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = starting_batch_size

            def is_oom_error(e):
                msg = str(e).lower()
                return (
                    isinstance(e, RuntimeError) and (
                        "mps backend out of memory" in msg or
                        "out of memory" in msg or
                        "cannot allocate memory" in msg or
                        "invalid buffer size" in msg or
                        "cpu memory" in msg
                    )
                )

            if auto_increment:
                # Try increasing batch sizes up to max_batch_size
                while True:
                    try:
                        result = func(*args, batch_size=batch_size, **kwargs)
                        batch_size *= 2
                        if max_batch_size and batch_size > max_batch_size:
                            return result
                    except Exception as e:
                        if is_oom_error(e):
                            batch_size //= 2
                            if batch_size < 1:
                                raise RuntimeError("No executable batch size found.")
                            logger.info(f"OOM at batch size {batch_size*2}, using {batch_size}")

                            return func(*args, batch_size=batch_size, **kwargs)
                        else:
                            logger.exception("Unexpected error during batch size search")
                            raise e
            else:
                # Try decreasing batch sizes from starting point
                while batch_size > 0:
                    try:
                        return func(*args, batch_size=batch_size, **kwargs)
                    except Exception as e:
                        if is_oom_error(e):
                            logger.info(f"OOM at batch size {batch_size}, trying smaller size...")
                            batch_size //= 2
                        else:
                            logger.exception("Unexpected error during batch size search")
                            raise e
                raise RuntimeError("No executable batch size found.")
        return wrapper
    return decorator

def cleanup_memory(device):
    import torch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

def get_closest_batch_size(n: Union[int, float]) -> int:
    """
    Returns the closest power of 2 to the given positive number.

    Args:
        n (int or float): The number to find the closest power of 2 for. Must be positive.

    Returns:
        int: The closest power of 2 to `n`. If equally close to two powers, returns the higher one.

    Raises:
        ValueError: If n is not a positive number.
        TypeError: If n is not a number.
    """
    if not isinstance(n, (int, float)):
        raise TypeError("Input must be an integer or float.")
    if n <= 0:
        raise ValueError("Input must be a positive number.")

    lower_exp = math.floor(math.log2(n))
    upper_exp = math.ceil(math.log2(n))

    lower_power = 2 ** lower_exp
    upper_power = 2 ** upper_exp

    # Choose whichever is closer (prefer lower if exactly in the middle)
    if abs(n - lower_power) <= abs(n - upper_power):
        return lower_power
    else:
        return upper_power
