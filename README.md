# llama3_rag_reranker

Minimal, faithful reconstruction of the code behind the IJACSA paper
**"Transforming LLMs into Efficient Cross-Encoders"**: a generative LLaMA-3 reranker
(LoRA, **4-bit quantization**) that replaces the cross-encoder in a RAG pipeline
(Chroma + dual retriever: vector store + BM25, ensembled), with RAGAS-based evaluation.

Adapted to run free and local on Apple silicon (MacBook Pro M4 Pro, 24 GB) via **MLX / mlx-lm**;
OpenAI is optional.

> Scaffold in progress — full scripts, configs, tests, CI, and runbook are being added.
> "Compression" in the paper is **4-bit quantization**, not distillation.
