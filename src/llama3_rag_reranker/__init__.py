"""llama3_rag_reranker.

Minimal, faithful reconstruction of the IJACSA paper
"Transforming LLMs into Efficient Cross-Encoders": a generative LLaMA-3 reranker
(LoRA, 4-bit *quantization*) that replaces the cross-encoder in a RAG pipeline
(Chroma + dual retriever: vector store + BM25, ensembled), evaluated with RAGAS.

Adapted to run free and local on Apple silicon via MLX. OpenAI is optional.

Note: the paper's "compression" is 4-bit quantization, not distillation. There is
no teacher/student. Only RAGAS metrics are implemented here (no MRR/NDCG).
"""

__version__ = "0.1.0"
