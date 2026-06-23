"""Minimal serving hook: an importable score function and a FastAPI /rerank route.

Kept intentionally small — it's a hook for later, not a production server. The
reranker is loaded once and reused.
"""

from __future__ import annotations

from .config import Config, load_config
from .reranker import Llama3Reranker, RerankResult


def score(reranker: Llama3Reranker, query: str, candidates: list[str], top_k: int | None = None) -> RerankResult:
    """Rerank ``candidates`` for ``query``. Pure function over a loaded reranker."""
    return reranker.rerank(query, candidates, top_k=top_k)


def build_reranker(config: Config) -> Llama3Reranker:
    return Llama3Reranker(
        model=config.get("model.base"),
        adapter_path=config.get("paths.adapter"),
        max_tokens=config.get("reranker.max_tokens", 128),
        use_mlx=config.get("model.use_mlx", True),
    )


def create_app(config_path: str = "configs/default.yaml"):
    """Create the FastAPI app. Imported lazily so the package doesn't require FastAPI."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    config = load_config(config_path)
    reranker = build_reranker(config)
    app = FastAPI(title="llama3_rag_reranker", version="0.1.0")

    class RerankRequest(BaseModel):
        query: str
        candidates: list[str]
        top_k: int | None = None

    class RerankResponse(BaseModel):
        order: list[int]
        passages: list[str]
        scores: list[float]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/rerank", response_model=RerankResponse)
    def rerank(req: RerankRequest) -> RerankResponse:
        result = score(reranker, req.query, req.candidates, req.top_k)
        return RerankResponse(order=result.order, passages=result.passages, scores=result.scores)

    return app
