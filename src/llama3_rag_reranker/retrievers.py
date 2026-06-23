"""Dual retriever: a Chroma vector-store retriever + a BM25 retriever, ensembled.

The ensemble merges the per-retriever rankings with reciprocal-rank fusion (RRF),
the same scheme used by LangChain's ``EnsembleRetriever``. We keep our own small
implementation so the merge is dependency-light and unit-testable. ``chromadb``
and the embedder are imported lazily; ``BM25Retriever`` and ``EnsembleRetriever``
are pure Python (rank_bm25) and run in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Doc]: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Sparse lexical retriever backed by rank_bm25 (pure Python, no Java)."""

    def __init__(self, docs: list[Doc]):
        from rank_bm25 import BM25Okapi

        self.docs = list(docs)
        self._tokenized = [_tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def from_texts(cls, texts: list[str], ids: list[str] | None = None) -> BM25Retriever:
        ids = ids or [str(i) for i in range(len(texts))]
        return cls([Doc(id=i, text=t) for i, t in zip(ids, texts, strict=True)])

    def retrieve(self, query: str, k: int) -> list[Doc]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in order[:k]]


class VectorStoreRetriever:
    """Dense retriever over a Chroma collection using a configured embedder."""

    def __init__(self, collection, embedder):
        self._collection = collection
        self._embedder = embedder

    def retrieve(self, query: str, k: int) -> list[Doc]:
        qvec = self._embedder.embed_query(query)
        res = self._collection.query(query_embeddings=[qvec], n_results=k)
        ids = res["ids"][0]
        texts = res["documents"][0]
        metas = res.get("metadatas", [[{}] * len(ids)])[0]
        return [
            Doc(id=i, text=t, metadata=m or {})
            for i, t, m in zip(ids, texts, metas, strict=True)
        ]


class EnsembleRetriever:
    """Combine retrievers with weighted reciprocal-rank fusion.

    For each retriever, a document at rank ``r`` (0-based) contributes
    ``weight / (rrf_k + r + 1)`` to its fused score. Documents are de-duplicated
    by id and returned in descending fused-score order.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ):
        if not retrievers:
            raise ValueError("EnsembleRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights length must match retrievers length")
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[Doc]:
        fused: dict[str, float] = {}
        docs: dict[str, Doc] = {}
        # Pull a few extra per retriever so fusion has room to reorder.
        per_k = max(k, 1) * 2
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for rank, doc in enumerate(retriever.retrieve(query, per_k)):
                docs.setdefault(doc.id, doc)
                fused[doc.id] = fused.get(doc.id, 0.0) + weight / (self.rrf_k + rank + 1)
        ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)
        return [docs[i] for i in ranked_ids[:k]]
