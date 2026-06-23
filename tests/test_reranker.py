import pytest

from llama3_rag_reranker.prompts import (
    format_completion,
    parse_ranking,
    parse_ranking_with_coverage,
)
from llama3_rag_reranker.reranker import (
    Llama3Reranker,
    RerankerFallbackError,
    lexical_order,
)


def test_format_parse_round_trip():
    passages = ["a", "b", "c", "d"]
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        completion = format_completion(order)
        assert parse_ranking(completion, len(passages)) == order


def test_parse_ranking_strips_prompt_and_prose():
    text = "Some reasoning here.\nRanking: 2, 1, 3"
    assert parse_ranking(text, 3) == [1, 0, 2]


def test_parse_ranking_fills_missing_and_drops_out_of_range():
    # Only mentions 3 (->index 2) and an out-of-range 9; rest appended in order.
    assert parse_ranking("Ranking: 3, 9", 3) == [2, 0, 1]


def test_parse_ranking_dedups():
    assert parse_ranking("1, 1, 2", 3) == [0, 1, 2]


def test_lexical_order_prefers_overlap():
    query = "largest planet jupiter"
    passages = ["Mercury is the smallest planet.", "Jupiter is the largest planet."]
    assert lexical_order(query, passages)[0] == 1


def test_parse_ranking_with_coverage_reports_partial():
    # Full coverage when all ranks present.
    order, n_parsed = parse_ranking_with_coverage("Ranking: 2, 1, 3", 3)
    assert order == [1, 0, 2] and n_parsed == 3
    # Partial: only one valid index emitted -> coverage 1, rest back-filled.
    order, n_parsed = parse_ranking_with_coverage("Ranking: 3", 3)
    assert order == [2, 0, 1] and n_parsed == 1
    # Garbage -> zero coverage.
    _, n_parsed = parse_ranking_with_coverage("no numbers here", 3)
    assert n_parsed == 0


def test_reranker_fallback_without_mlx_sets_flag():
    # use_mlx=False forces the deterministic lexical path — no model needed.
    reranker = Llama3Reranker(model="unused", use_mlx=False)
    passages = ["Mercury is closest to the Sun.", "Jupiter is the largest planet."]
    result = reranker.rerank("which planet is largest", passages, top_k=1)
    assert result.passages == ["Jupiter is the largest planet."]
    assert len(result.order) == 1 and len(result.scores) == 1
    # The fallback must be observable, not silent.
    assert reranker.used_fallback is True
    assert reranker.fallback_reason is not None


def test_reranker_strict_raises_instead_of_falling_back():
    reranker = Llama3Reranker(model="unused", use_mlx=False, strict=True)
    with pytest.raises(RerankerFallbackError):
        reranker.rerank("q", ["a", "b"])
