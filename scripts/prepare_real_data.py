"""Convert public datasets into the unified jsonl format ({train,val,test}.jsonl).

    python scripts/prepare_real_data.py hatexplain --dataset-json HateXplain/Data/dataset.json \
        --divisions-json HateXplain/Data/post_id_divisions.json --out data/hatexplain
    python scripts/prepare_real_data.py memes --root hateful_memes --out data/hateful_memes
    python scripts/prepare_real_data.py csv --path hinglish.csv --text-col text --label-col label \
        --positive 1 --lang hien --out data/hinglish

The resulting directories work with scripts/train_real.py and the llm/ scripts.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rxhate.data import loaders  # noqa: E402


def write(out, splits):
    os.makedirs(out, exist_ok=True)
    for s, recs in splits.items():
        loaders.add_counterfactuals(recs)
        for r in recs:
            r["split"] = s
        with open(os.path.join(out, f"{s}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(s, len(recs), f"{sum(r['label'] for r in recs) / max(1, len(recs)):.1%} hateful")


def random_split(recs, seed=0, frac=(0.7, 0.1, 0.2)):
    random.Random(seed).shuffle(recs)
    a, b = int(frac[0] * len(recs)), int((frac[0] + frac[1]) * len(recs))
    return {"train": recs[:a], "val": recs[a:b], "test": recs[b:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["hatexplain", "memes", "hasoc", "csv"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-json")
    ap.add_argument("--divisions-json")
    ap.add_argument("--offensive-as", type=int, default=None)
    ap.add_argument("--root")
    ap.add_argument("--path")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive", nargs="+", default=["1"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--delimiter", default=",")
    a = ap.parse_args()
    if a.source == "hatexplain":
        write(a.out, loaders.load_hatexplain(a.dataset_json, a.divisions_json, a.offensive_as))
    elif a.source == "memes":
        splits = {"train": loaders.load_hateful_memes(a.root, "train.jsonl"),
                  "val": loaders.load_hateful_memes(a.root, "dev_seen.jsonl"),
                  "test": loaders.load_hateful_memes(a.root, "test_seen.jsonl")}
        write(a.out, splits)
    elif a.source == "hasoc":
        write(a.out, random_split(loaders.load_hasoc(a.path, lang=a.lang)))
    else:
        write(a.out, random_split(loaders.load_csv(a.path, a.text_col, a.label_col, a.positive, a.lang, a.delimiter)))
