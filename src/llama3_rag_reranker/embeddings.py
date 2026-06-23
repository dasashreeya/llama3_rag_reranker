"""Text embeddings: local sentence-transformers (default) or OpenAI (optional).

Default is BAAI/bge-small-en-v1.5 on MPS (free, local). Set ``backend: openai``
plus ``OPENAI_API_KEY`` to use OpenAI embeddings (faithful to the paper).
Heavy backends are imported lazily.
"""

from __future__ import annotations

import os
from typing import Protocol

from .config import resolve_device


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str = "auto"):
        self.model_name = model
        self.device = resolve_device(device)
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model_name = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder(config) -> Embedder:
    backend = config.get("embeddings.backend", "local")
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "embeddings.backend=openai but OPENAI_API_KEY is not set"
            )
        return OpenAIEmbedder(config.get("embeddings.openai_model", "text-embedding-3-small"))
    return LocalEmbedder(
        config.get("embeddings.local_model", "BAAI/bge-small-en-v1.5"),
        config.get("embeddings.device", "auto"),
    )
