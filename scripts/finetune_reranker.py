#!/usr/bin/env python
"""LoRA fine-tune the LLaMA-3 reranker with mlx-lm. See `l3rr-finetune`."""
from llama3_rag_reranker.scripts_entry import finetune_main

if __name__ == "__main__":
    finetune_main()
