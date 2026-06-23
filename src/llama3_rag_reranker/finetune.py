"""LoRA fine-tuning of a 4-bit Llama-3 base with mlx-lm.

This is the MLX (Apple-silicon) replacement for the paper's CUDA-only stack. The
compression is 4-bit *quantization* of the base model (e.g.
``mlx-community/Llama-3.2-3B-Instruct-4bit``); LoRA adapters are trained on top.
No distillation, no teacher/student.

We shell out to ``python -m mlx_lm lora`` so we use mlx-lm's maintained trainer
directly. MLX is Apple-only, so this step never runs in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import Config


def build_lora_command(config: Config) -> list[str]:
    data_dir = config.get("paths.data_dir", "data/rerank")
    adapter_path = config.get("paths.adapter", "models/adapter")
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(config.get("model.base")),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(config.get("finetune.iters", 200)),
        "--batch-size",
        str(config.get("finetune.batch_size", 1)),
        "--num-layers",
        str(config.get("finetune.num_layers", 8)),
        "--learning-rate",
        str(config.get("finetune.learning_rate", 1e-4)),
        "--max-seq-length",
        str(config.get("finetune.max_seq_length", 2048)),
        "--steps-per-eval",
        str(config.get("finetune.steps_per_eval", 50)),
        "--seed",
        str(config.get("seed", 13)),
    ]


def finetune(config: Config) -> dict:
    """Run mlx-lm LoRA training. Returns a summary dict with the adapter path."""
    cmd = build_lora_command(config)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {
        "adapter_path": config.get("paths.adapter", "models/adapter"),
        "base_model": config.get("model.base"),
    }
