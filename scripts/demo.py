"""Explain the trained policy's decision on your own post.

    python scripts/demo.py --text "ye log desh ke liye khatra hain" --parent "a new family moved in" \
        --image data/synhatemm/images/p00012.png
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rxhate import xai  # noqa: E402
from rxhate.data.synthetic import load_split  # noqa: E402
from rxhate.features import MultimodalFeaturizer  # noqa: E402
from rxhate.text import tokenize  # noqa: E402
from rxhate.vision import batch_image_features  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--parent", default="")
    ap.add_argument("--image", default=None)
    ap.add_argument("--community", default="news")
    ap.add_argument("--model", default="results/models/grpo_s0.pkl")
    ap.add_argument("--data", default="data/synhatemm")
    a = ap.parse_args()
    m = pickle.load(open(a.model, "rb"))
    F = MultimodalFeaturizer()
    F.fit(batch_image_features([], os.path.join(a.data, "vfeat_train.npy")))  # restore image standardisation
    rec = {"id": "demo", "tokens": tokenize(a.text), "rationale": [0] * len(tokenize(a.text)), "label": 0,
           "parent_text": a.parent or "none", "community": a.community, "thread_toxicity": 0.3,
           "image_path": a.image, "cf_tokens": None}
    B, _ = F.build([rec], "")
    text, _ = xai.explain(m, F, B, 0)
    print(text)
