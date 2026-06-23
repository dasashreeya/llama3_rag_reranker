import json

from llama3_rag_reranker.data import build_dataset, make_records


def _seed_rows():
    return [
        {
            "query": "Which planet is largest?",
            "candidates": ["Jupiter is largest.", "Mercury is smallest.", "Saturn has rings."],
            "gold_order": [0, 2, 1],
        },
        {
            "query": "Which planet is closest to the Sun?",
            "candidates": ["Mercury is closest.", "Earth supports life."],
            "gold_order": [0, 1],
        },
    ]


def test_make_records_shapes():
    records = make_records(_seed_rows())
    assert len(records) == 2
    for rec in records:
        assert set(rec.keys()) == {"prompt", "completion"}
        assert isinstance(rec["prompt"], str) and rec["prompt"]
        assert isinstance(rec["completion"], str) and rec["completion"]


def test_make_records_rejects_bad_gold_order():
    import pytest

    bad = [{"query": "q", "candidates": ["a", "b"], "gold_order": [0, 0]}]
    with pytest.raises(ValueError):
        make_records(bad)


def test_build_dataset_writes_train_valid(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    with open(seed_path, "w") as fh:
        for row in _seed_rows():
            fh.write(json.dumps(row) + "\n")

    summary = build_dataset(seed_path, tmp_path / "out", val_fraction=0.5, seed=7)

    assert summary["n_total"] == 2
    assert summary["n_train"] + summary["n_valid"] == 2
    assert summary["n_train"] >= 1

    for key in ("train_path", "valid_path"):
        rows = [json.loads(line) for line in open(summary[key]) if line.strip()]
        for row in rows:
            assert set(row.keys()) == {"prompt", "completion"}
