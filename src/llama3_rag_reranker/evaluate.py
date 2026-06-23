"""RAGAS evaluation: cross-encoder RAG pipeline vs fine-tuned LLaMA-3 reranker.

Reproduces the paper's comparison (its Table 2) using *only* RAGAS metrics:
answer relevancy, context precision, answer similarity, answer correctness.
No MRR/NDCG; no hardcoded numbers.

RAGAS needs a judge LLM + embeddings. If none is configured (no OPENAI_API_KEY),
metric computation is skipped and the results file is written with the table left
as TODO — the pipeline still runs end to end and the generated answers/contexts
are saved for inspection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config
from .rag import build_pipeline

# Metric names match ragas 0.2.x and the paper's Table 2.
METRIC_NAMES = [
    "answer_relevancy",
    "context_precision",
    "answer_similarity",
    "answer_correctness",
]

PIPELINES = {
    "cross_encoder": "Cross-encoder RAG (baseline)",
    "llama3": "Fine-tuned LLaMA-3 reranker",
}


def read_eval_set(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"question": obj["question"], "ground_truth": obj["ground_truth"]})
    return rows


def run_pipeline_over_queries(config: Config, mode: str, eval_rows: list[dict]) -> tuple[list[dict], object]:
    pipeline = build_pipeline(config, reranker_mode=mode)
    samples = []
    for row in eval_rows:
        out = pipeline.answer(row["question"])
        samples.append(
            {
                "user_input": row["question"],
                "response": out["answer"],
                "retrieved_contexts": out["contexts"],
                "reference": row["ground_truth"],
            }
        )
    return samples, pipeline


def judge_available(config: Config) -> bool:
    backend = config.get("judge.backend", "auto")
    if backend == "none":
        return False
    if backend in {"auto", "openai"}:
        return bool(os.environ.get("OPENAI_API_KEY"))
    return False


def _ragas_metrics():
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
    )

    return [answer_relevancy, context_precision, answer_similarity, answer_correctness]


def score_with_ragas(config: Config, samples: list[dict]) -> dict[str, float]:
    """Run RAGAS with an OpenAI judge. Returns {metric: mean_score}."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    model = config.get("judge.openai_model", "gpt-4o")
    llm = LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=_ragas_metrics(), llm=llm, embeddings=embeddings)
    df = result.to_pandas()
    return {m: float(df[m].mean()) for m in METRIC_NAMES if m in df.columns}


def _write_dataset(out_dir: Path, mode: str, samples: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_dataset_{mode}.jsonl", "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


def _results_table(scores: dict[str, dict[str, float]]) -> str:
    header = "| Pipeline | " + " | ".join(METRIC_NAMES) + " |\n"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 1) + "\n"
    rows = ""
    for mode, label in PIPELINES.items():
        cells = " | ".join(f"{scores[mode][m]:.4f}" for m in METRIC_NAMES)
        rows += f"| {label} | {cells} |\n"
    return header + sep + rows


def _todo_table() -> str:
    header = "| Pipeline | " + " | ".join(METRIC_NAMES) + " |\n"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 1) + "\n"
    rows = "".join(
        f"| {label} | " + " | ".join(["TODO"] * len(METRIC_NAMES)) + " |\n"
        for label in PIPELINES.values()
    )
    return header + sep + rows


def run_evaluation(config: Config) -> dict:
    """Run both pipelines over the eval set and write the results file.

    Returns a summary dict. Computes RAGAS scores only if a judge is configured.
    """
    eval_rows = read_eval_set(config.get("paths.eval"))
    results_path = Path(config.get("paths.results", "data/results.md"))
    dataset_dir = results_path.parent / "eval_datasets"

    samples_by_mode: dict[str, list[dict]] = {}
    approximate_generator = False
    reranker_fallback: tuple[bool, str | None] = (False, None)
    for mode in PIPELINES:
        samples, pipeline = run_pipeline_over_queries(config, mode, eval_rows)
        samples_by_mode[mode] = samples
        _write_dataset(dataset_dir, mode, samples)
        # The LLaMA-3 arm is the one under test: flag if it silently used the fallback.
        if mode == "llama3" and getattr(pipeline.reranker, "used_fallback", False):
            reranker_fallback = (True, getattr(pipeline.reranker, "fallback_reason", None))

    # Reflect whether the generator that produced these answers was a substitute.
    from .generator import get_generator

    approximate_generator = get_generator(config).approximate

    # If the LLaMA-3 arm silently used the lexical fallback, the comparison is
    # scoring the fallback, not the fine-tuned model. Make that impossible to miss.
    fallback_banner = ""
    if reranker_fallback[0]:
        fallback_banner = (
            "> ⚠️ **Invalid comparison:** the LLaMA-3 reranker fell back to lexical "
            f"ordering ({reranker_fallback[1]}); these results do NOT reflect the "
            "fine-tuned model. Fix the MLX model/adapter (or set `reranker.strict=true` "
            "to fail fast) and re-run.\n\n"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    if judge_available(config):
        scores = {m: score_with_ragas(config, samples_by_mode[m]) for m in PIPELINES}
        table = _results_table(scores)
        note = ""
        if approximate_generator:
            note = (
                "\n> **Note:** answers were produced by an approximate (non-gpt-4o) "
                "generator; treat these numbers as indicative, not the paper's.\n"
            )
        body = f"# RAGAS Results\n\n{fallback_banner}{table}{note}"
        computed = True
    else:
        scores = {}
        body = (
            "# RAGAS Results\n\n"
            f"{fallback_banner}"
            "RAGAS metric computation was **skipped**: no judge LLM is configured "
            "(set `OPENAI_API_KEY`, or run on a backend with a judge). The pipeline "
            "ran end to end and the generated answers/contexts are saved under "
            f"`{dataset_dir}/` for inspection.\n\n"
            "<!-- TODO: run with a judge LLM to fill in the table below. Do not hand-enter numbers. -->\n\n"
            + _todo_table()
        )
        computed = False

    results_path.write_text(body)
    return {
        "results_path": str(results_path),
        "dataset_dir": str(dataset_dir),
        "ragas_computed": computed,
        "approximate_generator": approximate_generator,
        "reranker_used_fallback": reranker_fallback[0],
        "reranker_fallback_reason": reranker_fallback[1],
        "n_queries": len(eval_rows),
        "scores": scores,
    }
