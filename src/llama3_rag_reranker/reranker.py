"""Generative LLaMA-3 reranker (MLX) with a deterministic lexical fallback.

The reranker replaces the cross-encoder: it takes a query and candidate passages
and returns them reordered, best first. Fine-tuning is LoRA over a 4-bit
*quantized* Llama-3 base (quantization, not distillation — no teacher/student).

``mlx_lm`` is imported lazily so this module stays importable on non-Apple CI,
where only the prompt/parse round-trip is exercised.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .prompts import format_rerank_prompt, parse_ranking_with_coverage

logger = logging.getLogger(__name__)


class RerankerFallbackError(RuntimeError):
    """Raised in strict mode when the reranker cannot use the fine-tuned model."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_order(query: str, passages: list[str]) -> list[int]:
    """Deterministic fallback ranking by query/passage token overlap.

    Used when MLX is unavailable (e.g. CI, or before a model is downloaded) so
    the pipeline and smoke test still run end to end. Ties keep original order.
    """
    q = set(_tokenize(query))
    scored = [
        (-(len(q & set(_tokenize(p)))), i)  # negative overlap -> descending
        for i, p in enumerate(passages)
    ]
    scored.sort()
    return [i for _, i in scored]


@dataclass
class RerankResult:
    order: list[int]            # original indices, best first
    passages: list[str]         # passages in ranked order
    scores: list[float]         # rank-derived scores in [0, 1], best first


def _rank_scores(n: int) -> list[float]:
    if n <= 1:
        return [1.0] * n
    return [1.0 - i / (n - 1) for i in range(n)]


class Llama3Reranker:
    """Loads a base 4-bit Llama-3 model + LoRA adapter and reranks via generation."""

    def __init__(
        self,
        model: str,
        adapter_path: str | None = None,
        max_tokens: int = 128,
        use_mlx: bool = True,
        strict: bool = False,
    ):
        self.model_name = model
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.use_mlx = use_mlx
        # strict=True: hard-fail instead of silently scoring the lexical fallback.
        # Recommended for the real RAGAS comparison (see README).
        self.strict = strict
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None
        # Public flags so callers (e.g. evaluation) can detect a tainted run.
        self.used_fallback = False
        self.fallback_reason: str | None = None
        self._warned_fallback = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            from mlx_lm import load
        except ImportError as exc:
            self._load_error = f"mlx_lm not importable ({exc})"
            return False
        try:
            self._model, self._tokenizer = load(
                self.model_name, adapter_path=self.adapter_path
            )
        except Exception as exc:  # noqa: BLE001 - surface any load failure as a fallback
            self._load_error = f"failed to load {self.model_name!r}: {exc}"
            return False
        return True

    def _generate(self, prompt: str) -> str:
        from mlx_lm import generate

        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

    def _fallback(self, query: str, passages: list[str], reason: str, loud: bool) -> list[int]:
        """Record and (loudly) report that the fine-tuned model was NOT used."""
        self.used_fallback = True
        self.fallback_reason = reason
        if self.strict:
            raise RerankerFallbackError(
                f"reranker fell back to lexical ordering: {reason}. "
                "Set reranker.strict=false to allow the fallback."
            )
        if not self._warned_fallback:
            log = logger.warning if loud else logger.info
            log(
                "LLaMA-3 reranker NOT used (%s). Falling back to deterministic "
                "lexical ordering; results do NOT reflect the fine-tuned model. "
                "Set reranker.strict=true to fail instead.",
                reason,
            )
            self._warned_fallback = True
        return lexical_order(query, passages)

    def order_for(self, query: str, passages: list[str]) -> list[int]:
        if not passages:
            return []
        if not self.use_mlx:
            # Explicitly disabled (e.g. offline/unit tests): expected, logged quietly.
            return self._fallback(query, passages, "MLX disabled (use_mlx=false)", loud=False)
        if not self._ensure_loaded():
            # Wanted the model but it is unavailable: this is the dangerous case.
            return self._fallback(query, passages, self._load_error or "MLX load failed", loud=True)

        prompt = format_rerank_prompt(query, passages)
        text = self._generate(prompt)
        order, n_parsed = parse_ranking_with_coverage(text, len(passages))
        if n_parsed < len(passages):
            msg = (
                f"reranker output parsed only {n_parsed}/{len(passages)} valid ranks; "
                f"{len(passages) - n_parsed} back-filled in original order"
            )
            if self.strict:
                raise RerankerFallbackError(f"{msg}. Raw output: {text[:200]!r}")
            logger.warning("LLaMA-3 reranker: %s. Raw output: %r", msg, text[:200])
        return order

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> RerankResult:
        order = self.order_for(query, passages)
        ranked = [passages[i] for i in order]
        scores = _rank_scores(len(order))
        if top_k is not None:
            order, ranked, scores = order[:top_k], ranked[:top_k], scores[:top_k]
        return RerankResult(order=order, passages=ranked, scores=scores)


class CrossEncoderReranker:
    """Baseline cross-encoder reranker (the component the paper replaces)."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> RerankResult:
        if not passages:
            return RerankResult([], [], [])
        self._ensure_loaded()
        raw = self._model.predict([(query, p) for p in passages])
        order = sorted(range(len(passages)), key=lambda i: float(raw[i]), reverse=True)
        ranked = [passages[i] for i in order]
        scores = [float(raw[i]) for i in order]
        if top_k is not None:
            order, ranked, scores = order[:top_k], ranked[:top_k], scores[:top_k]
        return RerankResult(order=order, passages=ranked, scores=scores)
