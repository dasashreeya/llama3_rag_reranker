"""Build the reranking fine-tuning dataset for mlx-lm LoRA.

Reads a tiny seed file of labelled reranking examples (query + candidate passages
+ gold order) and emits ``train.jsonl`` / ``valid.jsonl`` in mlx-lm's
prompt/completion format:

    {"prompt": "<rerank prompt>", "completion": " 3, 1, 2"}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .prompts import format_completion, format_rerank_prompt


def _read_seed(seed_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(seed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_records(seed_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert seed rows into mlx-lm prompt/completion records."""
    records: list[dict[str, str]] = []
    for row in seed_rows:
        query = row["query"]
        candidates = row["candidates"]
        order = row["gold_order"]
        if sorted(order) != list(range(len(candidates))):
            raise ValueError(
                f"gold_order {order} is not a permutation of range({len(candidates)})"
            )
        records.append(
            {
                "prompt": format_rerank_prompt(query, candidates),
                "completion": format_completion(order),
            }
        )
    return records


def build_dataset(
    seed_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> dict[str, Any]:
    """Build train/valid jsonl from the seed file. Returns a summary dict."""
    rows = _read_seed(seed_path)
    records = make_records(rows)

    rng = random.Random(seed)
    rng.shuffle(records)

    n_val = max(1, int(round(len(records) * val_fraction))) if len(records) > 1 else 0
    valid = records[:n_val]
    train = records[n_val:] or records  # never leave train empty

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)

    return {
        "n_total": len(records),
        "n_train": len(train),
        "n_valid": len(valid),
        "train_path": str(out / "train.jsonl"),
        "valid_path": str(out / "valid.jsonl"),
    }


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
