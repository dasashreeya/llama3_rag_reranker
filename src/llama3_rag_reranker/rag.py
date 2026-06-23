"""RAG pipeline: corpus -> embeddings -> Chroma + BM25 -> ensemble -> rerank -> generate.

Faithful to the paper: documents are embedded and stored in Chroma; a dual
retriever (vector store + BM25) is ensembled; the candidates are reranked (by the
fine-tuned LLaMA-3 reranker, or the cross-encoder baseline); a generator LLM
produces the final answer from the top passages.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .embeddings import get_embedder
from .generator import get_generator
from .reranker import CrossEncoderReranker, Llama3Reranker
from .retrievers import BM25Retriever, Doc, EnsembleRetriever, VectorStoreRetriever


def load_corpus(corpus_dir: str | Path) -> list[Doc]:
    """Load one document per ``*.txt`` file under ``corpus_dir`` (sorted)."""
    docs: list[Doc] = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text = path.read_text().strip()
        if text:
            docs.append(Doc(id=path.stem, text=text, metadata={"source": path.name}))
    return docs


def build_chroma(docs: list[Doc], embedder, persist_dir: str | Path):
    """(Re)build a persistent Chroma collection from docs and return it."""
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    name = "corpus"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)
    embeddings = embedder.embed_documents([d.text for d in docs])
    collection.add(
        ids=[d.id for d in docs],
        documents=[d.text for d in docs],
        embeddings=embeddings,
        metadatas=[d.metadata for d in docs],
    )
    return collection


def make_reranker(config: Config):
    mode = config.get("reranker.mode", "llama3")
    if mode == "cross_encoder":
        return CrossEncoderReranker(
            config.get("reranker.cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
    return Llama3Reranker(
        model=config.get("model.base"),
        adapter_path=config.get("paths.adapter"),
        max_tokens=config.get("reranker.max_tokens", 128),
        use_mlx=config.get("model.use_mlx", True),
        strict=config.get("reranker.strict", False),
    )


class RagPipeline:
    def __init__(self, config: Config, ensemble: EnsembleRetriever, reranker, generator):
        self.config = config
        self.ensemble = ensemble
        self.reranker = reranker
        self.generator = generator
        self.candidates_k = config.get("retrieval.candidates_k", 8)
        self.top_k = config.get("retrieval.top_k", 4)

    def retrieve_and_rerank(self, query: str) -> list[str]:
        candidates = self.ensemble.retrieve(query, self.candidates_k)
        texts = [d.text for d in candidates]
        result = self.reranker.rerank(query, texts, top_k=self.top_k)
        return result.passages

    def answer(self, query: str) -> dict:
        contexts = self.retrieve_and_rerank(query)
        answer = self.generator.generate(query, contexts)
        return {"query": query, "answer": answer, "contexts": contexts}


def build_pipeline(config: Config, reranker_mode: str | None = None) -> RagPipeline:
    """Assemble the full pipeline. ``reranker_mode`` overrides the config (used by eval)."""
    if reranker_mode is not None:
        config.set("reranker.mode", reranker_mode)

    embedder = get_embedder(config)
    docs = load_corpus(config.get("paths.corpus"))
    if not docs:
        raise RuntimeError(f"no documents found under {config.get('paths.corpus')}")

    collection = build_chroma(docs, embedder, config.get("paths.chroma"))
    vector = VectorStoreRetriever(collection, embedder)
    bm25 = BM25Retriever(docs)
    ensemble = EnsembleRetriever(
        [vector, bm25],
        weights=[
            config.get("retrieval.vector_weight", 0.5),
            config.get("retrieval.bm25_weight", 0.5),
        ],
        rrf_k=config.get("retrieval.rrf_k", 60),
    )
    reranker = make_reranker(config)
    generator = get_generator(config)
    return RagPipeline(config, ensemble, reranker, generator)
