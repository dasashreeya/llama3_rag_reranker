# llama3_rag_reranker

[![CI](https://github.com/dasashreeya/llama3_rag_reranker/actions/workflows/ci.yml/badge.svg)](https://github.com/dasashreeya/llama3_rag_reranker/actions/workflows/ci.yml)

Minimal, faithful reconstruction of the code behind the IJACSA paper
**"Transforming LLMs into Efficient Cross-Encoders"** — a generative **LLaMA-3 reranker**
(LoRA fine-tuned over a **4-bit quantized** base) that replaces the cross-encoder in a RAG
pipeline, evaluated with **RAGAS**.

> **Honest note:** the paper's "compression" is **4-bit quantization** of the base model, not
> distillation — there is no teacher/student. Only RAGAS metrics are implemented here (the paper
> reported MRR/NDCG in its abstract but only computed RAGAS); no MRR/NDCG, and no numbers are hardcoded.

Paper: _Transforming LLMs into Efficient Cross-Encoders_, International Journal of Advanced
Computer Science and Applications (IJACSA). <!-- TODO: add DOI / URL -->

## What it does

```
documents ─► embeddings ─► Chroma  ┐
                                    ├─► ensemble (RRF) ─► LLaMA-3 reranker ─► generator LLM ─► answer
queries ──────────────► BM25 ───────┘
```

- **Fine-tune** a LLaMA-3 reranker with LoRA on a small reranking dataset (query + candidate
  passages in, gold ranking out), using **MLX / mlx-lm** on Apple silicon.
- **RAG pipeline:** load documents, embed them, store in **Chroma**; a **dual retriever**
  (vector store + BM25) is **ensembled**; the fine-tuned reranker reorders the candidates; a
  **generator LLM** produces the final answer.
- **Evaluate** with RAGAS (answer relevancy, context precision, answer similarity, answer
  correctness), comparing a **cross-encoder** RAG pipeline against the **LLaMA-3 reranker** pipeline
  (the paper's Table 2).

This is a faithful **Mac adaptation** of the paper's CUDA-only stack (the original used
Unsloth/bitsandbytes; here it's MLX). Target hardware: MacBook Pro M4 Pro, 24 GB.

## Free vs. API key

| Step | Free & local | Needs `OPENAI_API_KEY` |
|------|--------------|------------------------|
| `build_data` | ✅ | — |
| `finetune_reranker` (MLX LoRA) | ✅ (Apple silicon) | — |
| Embeddings | ✅ `BAAI/bge-small-en-v1.5` on MPS | optional: OpenAI embeddings |
| Retrieval (Chroma + BM25 + ensemble) | ✅ | — |
| Reranking (LLaMA-3 / cross-encoder) | ✅ | — |
| Generator LLM | ⚠️ local Ollama or a **stub** (flagged approximate) | ✅ gpt-4o (faithful) |
| RAGAS metrics | ✗ (skipped, table left TODO) | ✅ gpt-4o judge (faithful) |

Without a key and without Ollama, the generator falls back to a **deterministic stub** and RAGAS
metric computation is **skipped** — the pipeline still runs end to end and the generated
answers/contexts are saved for inspection. The stub is clearly flagged and is **not** faithful;
configure a judge for real numbers.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1
# optional, for the faithful generator + RAGAS judge:
export OPENAI_API_KEY=sk-...
```

## Smoke test first

A tiny config (a few docs, a few queries, 4 LoRA steps) runs the whole pipeline end to end on
CPU/MPS with no API key. Run this before the full config:

```bash
l3rr-build-data --config configs/smoke.yaml
l3rr-finetune   --config configs/smoke.yaml      # downloads Llama-3.2-3B-Instruct-4bit (~1.8GB) on first run
l3rr-build-rag  --config configs/smoke.yaml
l3rr-evaluate   --config configs/smoke.yaml      # writes data/smoke/results.md (RAGAS TODO without a judge)
```

Then the full run uses `configs/default.yaml`.

## Reranker fallback (read before the real run)

If MLX or the LoRA adapter can't be loaded, the LLaMA-3 reranker falls back to a deterministic
**lexical** ordering so the pipeline still runs. This fallback is **never silent**: it logs a
loud `WARNING`, sets a flag that `evaluate` turns into a banner at the top of the results file,
and `l3rr-evaluate` prints a warning — so a RAGAS comparison can't accidentally score the
fallback instead of the fine-tuned model. For the real comparison, set `reranker.strict: true`
in your config to hard-fail instead of falling back. (A degenerate/unparseable model output is
flagged the same way, reporting how many ranks were back-filled.)

## MLX model switch

Default base model is `mlx-community/Llama-3.2-3B-Instruct-4bit` (fits 24 GB comfortably). To use
the larger Llama-3 8B, set in your config:

```yaml
model:
  base: mlx-community/Meta-Llama-3-8B-Instruct-4bit   # slower, memory-tight on 24GB
```

## Expected runtimes (M4 Pro, 24 GB; first run includes model downloads)

| Step | Smoke | Full (`default.yaml`) |
|------|-------|------------------------|
| `build_data` | seconds | seconds |
| `finetune_reranker` | ~1–3 min (mostly the ~1.8GB download) | tens of minutes |
| `build_rag` | seconds (after bge download) | seconds–minutes by corpus size |
| `evaluate_ragas` | ~1 min (stub, RAGAS skipped) | minutes (judge LLM calls) |

## Serving (hook for later)

```bash
l3rr-serve --config configs/default.yaml          # FastAPI: POST /rerank {query, candidates, top_k}
```
`llama3_rag_reranker.serve.score(reranker, query, candidates)` is the importable score function.

## Results

RAGAS comparison (cross-encoder RAG vs. LLaMA-3 reranker RAG) is produced by `l3rr-evaluate` with a
judge LLM configured, and written to the results file. **TODO: run and fill in** — numbers are not
hand-entered.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

CI (GitHub Actions) runs ruff + pytest on a slim, pure-Python dependency set — no MLX, no model
downloads (MLX is Apple-only; finetuning is a local step).
