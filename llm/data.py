"""Turn unified records (synthetic or real, see rxhate/data/loaders.py) into HF datasets."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rxhate.features import mentions_identity  # noqa: E402

from llm.prompts import build_messages, target_answer  # noqa: E402


def load_records(source, split):
    """source: a directory with {train,val,test}.jsonl in the unified schema
    (e.g. data/synhatemm, or one written by scripts/prepare_real_data.py)."""
    with open(os.path.join(source, f"{split}.jsonl")) as f:
        return [json.loads(line) for line in f]


def to_rows(records, captions=None, with_answer=False):
    rows = []
    for r in records:
        row = {
            "prompt": build_messages(r, captions),
            "label": int(r["label"]),
            "tokens": r["tokens"],
            "rationale": [int(x) for x in r["rationale"]],
            "has_gold_rationale": bool(r.get("has_gold_rationale", False)),
            "has_identity": bool(mentions_identity(r["tokens"])),
            "id": r["id"],
        }
        if with_answer:
            row["messages"] = row["prompt"] + [{"role": "assistant", "content": target_answer(r)}]
        rows.append(row)
    return rows


def hf_dataset(records, captions=None, with_answer=False):
    from datasets import Dataset
    return Dataset.from_list(to_rows(records, captions, with_answer))
