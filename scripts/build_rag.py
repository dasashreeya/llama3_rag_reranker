#!/usr/bin/env python
"""Build the RAG pipeline (Chroma + dual retriever). See `l3rr-build-rag`."""
from llama3_rag_reranker.scripts_entry import build_rag_main

if __name__ == "__main__":
    build_rag_main()
