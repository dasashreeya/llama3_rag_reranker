from llama3_rag_reranker.retrievers import (
    BM25Retriever,
    Doc,
    EnsembleRetriever,
)


def _corpus():
    return [
        Doc(id="jupiter", text="Jupiter is the largest planet in the Solar System."),
        Doc(id="mercury", text="Mercury is the smallest planet and closest to the Sun."),
        Doc(id="saturn", text="Saturn is best known for its bright ring system."),
    ]


def test_bm25_ranks_by_relevance():
    bm25 = BM25Retriever(_corpus())
    # BM25 is lexical (no stemming), so match the corpus wording: "ring system".
    ranked = bm25.retrieve("bright ring system", k=3)
    assert [d.id for d in ranked][0] == "saturn"
    assert len(ranked) == 3


def test_bm25_respects_k():
    bm25 = BM25Retriever(_corpus())
    assert len(bm25.retrieve("planet", k=2)) == 2


class _FakeRetriever:
    """Returns a fixed ranked list, ignoring the query (for ensemble tests)."""

    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query, k):
        return self._docs[:k]


def test_ensemble_merges_and_dedups():
    a = Doc(id="a", text="alpha")
    b = Doc(id="b", text="bravo")
    c = Doc(id="c", text="charlie")
    # Retriever 1 ranks a > b; retriever 2 ranks b > c. 'b' is shared.
    r1 = _FakeRetriever([a, b])
    r2 = _FakeRetriever([b, c])
    ensemble = EnsembleRetriever([r1, r2], weights=[1.0, 1.0], rrf_k=60)

    merged = ensemble.retrieve("q", k=3)
    ids = [d.id for d in merged]

    assert set(ids) == {"a", "b", "c"}      # union, de-duplicated
    assert len(ids) == len(set(ids))        # no duplicate 'b'
    assert ids[0] == "b"                    # appears in both -> highest fused score


def test_ensemble_weighting_changes_order():
    a = Doc(id="a", text="alpha")
    b = Doc(id="b", text="bravo")
    r1 = _FakeRetriever([a])      # only knows 'a'
    r2 = _FakeRetriever([b])      # only knows 'b'

    heavy_first = EnsembleRetriever([r1, r2], weights=[5.0, 1.0])
    assert heavy_first.retrieve("q", k=2)[0].id == "a"

    heavy_second = EnsembleRetriever([r1, r2], weights=[1.0, 5.0])
    assert heavy_second.retrieve("q", k=2)[0].id == "b"
