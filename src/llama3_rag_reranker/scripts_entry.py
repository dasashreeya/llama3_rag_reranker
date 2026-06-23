"""Console-script entry points. Each parses ``--config`` + ``--set k=v`` overrides."""

from __future__ import annotations

import argparse
import logging

from .config import load_config, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)


def _base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (dotted key), repeatable",
    )
    return p


def _load(args) -> object:
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.get("seed", 13))
    return cfg


def build_data_main() -> None:
    args = _base_parser("Build the reranking fine-tuning dataset").parse_args()
    cfg = _load(args)
    from .data import build_dataset

    summary = build_dataset(
        cfg.get("paths.rerank_seed"),
        cfg.get("paths.data_dir"),
        val_fraction=cfg.get("finetune.val_fraction", 0.2),
        seed=cfg.get("seed", 13),
    )
    print(f"Built dataset: {summary}")


def finetune_main() -> None:
    args = _base_parser("LoRA fine-tune the LLaMA-3 reranker (mlx-lm)").parse_args()
    cfg = _load(args)
    from .finetune import finetune

    print(f"Fine-tune complete: {finetune(cfg)}")


def build_rag_main() -> None:
    args = _base_parser("Build the RAG pipeline (Chroma + dual retriever)").parse_args()
    cfg = _load(args)
    from .rag import build_pipeline

    pipeline = build_pipeline(cfg)
    print(
        f"RAG pipeline ready: reranker={cfg.get('reranker.mode')}, "
        f"generator={pipeline.generator.backend} "
        f"(approximate={pipeline.generator.approximate}), "
        f"candidates_k={pipeline.candidates_k}, top_k={pipeline.top_k}"
    )


def evaluate_main() -> None:
    args = _base_parser("Evaluate both pipelines with RAGAS").parse_args()
    cfg = _load(args)
    from .evaluate import run_evaluation

    summary = run_evaluation(cfg)
    print(f"Evaluation complete: {summary}")
    if summary["reranker_used_fallback"]:
        print(
            "WARNING: the LLaMA-3 reranker used the lexical FALLBACK "
            f"({summary['reranker_fallback_reason']}) — this comparison does NOT "
            "reflect the fine-tuned model. Fix MLX/adapter or set reranker.strict=true."
        )
    if not summary["ragas_computed"]:
        print(
            "RAGAS metrics were skipped (no judge LLM). Results table left as TODO at "
            f"{summary['results_path']}."
        )


def serve_main() -> None:
    p = _base_parser("Serve the reranker (FastAPI /rerank)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    import uvicorn

    from .serve import create_app

    uvicorn.run(create_app(args.config), host=args.host, port=args.port)
