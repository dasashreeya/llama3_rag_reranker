#!/usr/bin/env python
"""Serve the reranker (FastAPI /rerank). See `l3rr-serve`."""
from llama3_rag_reranker.scripts_entry import serve_main

if __name__ == "__main__":
    serve_main()
