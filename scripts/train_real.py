"""Train + evaluate the full pipeline (SFT -> GRPO -> XAI) on a prepared real dataset.

    python scripts/train_real.py --data data/hatexplain --images-root data/hatexplain --out results/hatexplain
For Hateful Memes pass --images-root pointing at the folder that contains img/.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rxhate.evaluate import evaluate_policy  # noqa: E402
from rxhate.features import MultimodalFeaturizer  # noqa: E402
from rxhate.grpo import GRPOConfig, train_grpo  # noqa: E402
from rxhate.model import FusionPolicy, train_sft  # noqa: E402
from rxhate.rewards import RewardConfig  # noqa: E402


def load(d, s):
    with open(os.path.join(d, f"{s}.jsonl")) as f:
        return [json.loads(x) for x in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-epochs", type=int, default=15)
    ap.add_argument("--grpo-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, va, te = load(a.data, "train"), load(a.data, "val"), load(a.data, "test")
    root = a.images_root or a.data
    F = MultimodalFeaturizer()
    Btr, XVtr = F.build(tr, root)
    Bva, _ = F.build(va, root)
    Bte, XVte = F.build(te, root)
    Btr_cf, _ = F.build(tr, root, XV_raw=XVtr, use_cf=True)
    Bte_cf, _ = F.build(te, root, XV_raw=XVte, use_cf=True)
    cf_tr = np.array([r.get("cf_tokens") is not None for r in tr])
    cf_te = np.array([r.get("cf_tokens") is not None for r in te])
    mods = ("text", "image", "context") if any(r.get("image_path") for r in tr) else ("text", "context")
    m = FusionPolicy(Btr.XT.shape[1], Btr.XV.shape[1], Btr.XC.shape[1], Btr.XM.shape[1], F.rat.dim, seed=a.seed,
                     modalities=mods)
    train_sft(m, Btr, Bva, epochs=a.sft_epochs)
    res = {"sft": evaluate_policy(m, F, Bte, Bte_cf, cf_te)}
    pickle.dump(m, open(os.path.join(a.out, "sft.pkl"), "wb"))
    m, hist = train_grpo(m, m.clone(), Btr, Btr_cf, cf_tr, Bva, F, GRPOConfig(epochs=a.grpo_epochs, seed=a.seed),
                         RewardConfig())
    res["grpo"] = evaluate_policy(m, F, Bte, Bte_cf, cf_te)
    res["grpo_log"] = hist
    pickle.dump(m, open(os.path.join(a.out, "grpo.pkl"), "wb"))
    for k in ("sft", "grpo"):
        res[k].pop("_p1", None)
        print(k, json.dumps(res[k]["overall"]), "bias", json.dumps(res[k]["bias"]))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
