"""Reranking prompt formatting and ranking parsing.

The generative reranker is shown a query plus numbered candidate passages and is
asked to emit a ranking as a comma-separated list of 1-based indices, best first.
``format_completion`` and ``parse_ranking`` are exact round-trip inverses, which
is what the unit tests check.
"""

from __future__ import annotations

import re

RERANK_SYSTEM = (
    "You are a passage reranker. Given a query and a numbered list of candidate "
    "passages, order the passages from most to least relevant to the query. "
    "Respond with only the ranking as a comma-separated list of passage numbers, "
    "best first, e.g. 'Ranking: 3, 1, 2'."
)

_RANKING_PREFIX = "Ranking:"


def format_candidates(passages: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def format_rerank_prompt(query: str, passages: list[str]) -> str:
    """Build the full reranker prompt for a query and its candidate passages."""
    return (
        f"{RERANK_SYSTEM}\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{format_candidates(passages)}\n\n"
        f"{_RANKING_PREFIX}"
    )


def format_completion(order: list[int]) -> str:
    """Render a ranking (0-based original indices, best first) as a completion.

    Output uses 1-based indices to match the human-readable prompt, e.g.
    ``order=[2, 0, 1]`` -> ``" 3, 1, 2"`` (leading space separates it from the
    prompt's trailing ``Ranking:``).
    """
    return " " + ", ".join(str(i + 1) for i in order)


def extract_ranked_indices(text: str, n: int) -> list[int]:
    """Extract the valid, de-duplicated 0-based indices the model actually emitted.

    No back-filling: the returned list may be shorter than ``n`` (or empty) when
    the model's output was incomplete or unparseable. Callers use ``len(...)`` as
    a coverage signal to detect a degenerate ranking.
    """
    # Drop everything up to and including a 'Ranking:' marker if present.
    marker = text.lower().rfind(_RANKING_PREFIX.lower())
    if marker != -1:
        text = text[marker + len(_RANKING_PREFIX) :]

    seen: list[int] = []
    for tok in re.findall(r"\d+", text):
        idx = int(tok) - 1
        if 0 <= idx < n and idx not in seen:
            seen.append(idx)
        if len(seen) == n:
            break
    return seen


def parse_ranking_with_coverage(text: str, n: int) -> tuple[list[int], int]:
    """Like :func:`parse_ranking`, but also report how many ranks the model emitted.

    Returns ``(order, n_parsed)`` where ``order`` is a full permutation of
    ``range(n)`` and ``n_parsed`` is the number of valid indices found before any
    back-filling. ``n_parsed < n`` means the rest were appended in original order.
    """
    seen = extract_ranked_indices(text, n)
    n_parsed = len(seen)
    for idx in range(n):
        if idx not in seen:
            seen.append(idx)
    return seen, n_parsed


def parse_ranking(text: str, n: int) -> list[int]:
    """Parse a ranking string into 0-based original indices, best first.

    Robust to surrounding prose, a leading ``Ranking:`` token, duplicates, and
    out-of-range numbers. Any candidates the model omitted are appended in their
    original order so the result is always a full permutation of ``range(n)``.
    """
    order, _ = parse_ranking_with_coverage(text, n)
    return order
