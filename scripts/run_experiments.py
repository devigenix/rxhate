"""Run every experiment reported in the paper.

    python scripts/run_experiments.py --stage main       # baselines, SFT variants, GRPO (3 seeds)
    python scripts/run_experiments.py --stage ablation   # reward / optimiser ablations (needs main)
    python scripts/run_experiments.py --stage lowres     # low-resource Hindi curve
    python scripts/run_experiments.py --stage feedback   # human-rationale budget curve
    python scripts/run_experiments.py --stage xai        # explainer comparison + modality Shapley (needs main)

Results are written as JSON to results/; trained policies are pickled to results/models/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rxhate.baselines import multimodal_lr, tfidf_lr  # noqa: E402
from rxhate.data.synthetic import GenConfig, generate, load_split  # noqa: E402
from rxhate.evaluate import evaluate_policy, evaluate_probs  # noqa: E402
from rxhate.features import MultimodalFeaturizer  # noqa: E402
from rxhate.grpo import GRPOConfig, train_grpo  # noqa: E402
from rxhate.metrics import plausibility, topk_mask  # noqa: E402
from rxhate.model import FusionPolicy, train_sft  # noqa: E402
from rxhate.rewards import RewardConfig  # noqa: E402
from rxhate import xai  # noqa: E402

DATA = "data/synhatemm"
RES = "results"
MOD = os.path.join(RES, "models")
SEEDS = (0, 1, 2)
SFT_EPOCHS = 15
GRPO_EPOCHS = 3
if os.environ.get("SMOKE"):  # quick end-to-end check of the whole pipeline
    RES, SEEDS, SFT_EPOCHS, GRPO_EPOCHS = "results_smoke", (0,), 1, 1
    MOD = os.path.join(RES, "models")


def jdump(obj, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def setup(train_records=None):
    if not os.path.exists(os.path.join(DATA, "train.jsonl")):
        generate(GenConfig(out_dir=DATA))
    tr, va, te = [load_split(DATA, s) for s in ("train", "val", "test")]
    if train_records is not None:
        tr = train_records(tr)
    F = MultimodalFeaturizer()
    Btr, XVtr = F.build(tr, DATA, "train" if train_records is None else None,
                        XV_raw=None if train_records is None else _vfeat_subset(tr))
    Bva, _ = F.build(va, DATA, "val")
    Bte, XVte = F.build(te, DATA, "test")
    Btr_cf, _ = F.build(tr, DATA, XV_raw=XVtr, use_cf=True)
    Bte_cf, _ = F.build(te, DATA, XV_raw=XVte, use_cf=True)
    cf_tr = np.array([r["cf_tokens"] is not None for r in tr])
    cf_te = np.array([r["cf_tokens"] is not None for r in te])
    return dict(tr=tr, va=va, te=te, F=F, Btr=Btr, Bva=Bva, Bte=Bte, Btr_cf=Btr_cf, Bte_cf=Bte_cf,
                cf_tr=cf_tr, cf_te=cf_te)


_ALL_TRAIN_V = None


def _vfeat_subset(records):
    global _ALL_TRAIN_V
    from rxhate.vision import batch_image_features
    if _ALL_TRAIN_V is None:
        full = load_split(DATA, "train")
        _ALL_TRAIN_V = dict(zip([r["id"] for r in full],
                                batch_image_features([os.path.join(DATA, r["image_path"]) for r in full],
                                                     os.path.join(DATA, "vfeat_train.npy"))))
    return np.stack([_ALL_TRAIN_V[r["id"]] for r in records])


def new_policy(S, seed, modalities=("text", "image", "context")):
    B = S["Btr"]
    return FusionPolicy(B.XT.shape[1], B.XV.shape[1], B.XC.shape[1], B.XM.shape[1], S["F"].rat.dim,
                        seed=seed, modalities=modalities)


def strip(ev):
    return {k: v for k, v in ev.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------- stages
def stage_main(S):
    out = {"baselines": {}, "sft": {}, "grpo": {}, "logs": {}}
    tr, te = S["tr"], S["te"]
    t0 = time.time()
    for name, kw in [("TF-IDF word + LR", dict(analyzer="word", ngram=(1, 2))),
                     ("TF-IDF char + LR", dict(analyzer="char_wb", ngram=(2, 5)))]:
        p, (vec, clf) = tfidf_lr(tr, te, **kw)
        pcf, _ = tfidf_lr(tr, [dict(r, tokens=r["cf_tokens"] or r["tokens"]) for r in te], **kw)
        out["baselines"][name] = strip(evaluate_probs(te, S["Bte"].y, p, pcf, S["cf_te"]))
        print(name, out["baselines"][name]["overall"], flush=True)
    p, clf = multimodal_lr(S["Btr"], S["Bte"])
    pcf, _ = multimodal_lr(S["Btr"], S["Bte_cf"])
    out["baselines"]["Early-fusion LR (T+V+C)"] = strip(evaluate_probs(te, S["Bte"].y, p, pcf, S["cf_te"]))
    print("MM-LR", out["baselines"]["Early-fusion LR (T+V+C)"]["overall"], flush=True)

    for mods, tag in [(("text",), "T"), (("text", "image"), "T+V"), (("text", "image", "context"), "T+V+C")]:
        out["sft"][tag] = []
        for seed in SEEDS:
            m = new_policy(S, seed, mods)
            log = []
            train_sft(m, S["Btr"], S["Bva"], epochs=SFT_EPOCHS, verbose=False, log=log)
            ev = evaluate_policy(m, S["F"], S["Bte"], S["Bte_cf"], S["cf_te"])
            out["sft"][tag].append(strip(ev))
            out["logs"][f"sft_{tag}_s{seed}"] = log
            print(f"SFT {tag} seed {seed}: {ev['overall']['macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if tag == "T+V+C":
                os.makedirs(MOD, exist_ok=True)
                pickle.dump(m, open(os.path.join(MOD, f"sft_s{seed}.pkl"), "wb"))
                np.save(os.path.join(RES, f"p1_sft_s{seed}.npy"), np.array(ev["_p1"]))
        jdump(out, "main.json")

    out["grpo"]["T+V+C"] = []
    for seed in SEEDS:
        m = pickle.load(open(os.path.join(MOD, f"sft_s{seed}.pkl"), "rb"))
        ref = m.clone()
        m, hist = train_grpo(m, ref, S["Btr"], S["Btr_cf"], S["cf_tr"], S["Bva"], S["F"],
                             GRPOConfig(epochs=GRPO_EPOCHS, seed=seed), RewardConfig(), verbose=True)
        ev = evaluate_policy(m, S["F"], S["Bte"], S["Bte_cf"], S["cf_te"])
        out["grpo"]["T+V+C"].append(strip(ev))
        out["logs"][f"grpo_s{seed}"] = hist
        pickle.dump(m, open(os.path.join(MOD, f"grpo_s{seed}.pkl"), "wb"))
        np.save(os.path.join(RES, f"p1_grpo_s{seed}.npy"), np.array(ev["_p1"]))
        print(f"GRPO seed {seed}: {ev['overall']['macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        jdump(out, "main.json")
    out["config"] = {"grpo": GRPOConfig(epochs=GRPO_EPOCHS).to_dict(), "reward": RewardConfig().to_dict(),
                     "sft_epochs": SFT_EPOCHS, "grpo_epochs": GRPO_EPOCHS, "seeds": SEEDS}
    jdump(out, "main.json")


ABLATIONS = {
    "Full (GRPO)": (dict(), dict()),
    "- fairness reward": (dict(use_fair=False), dict()),
    "- faithfulness reward": (dict(use_faith=False), dict()),
    "- plausibility reward": (dict(use_plaus=False), dict()),
    "- counterfactual pairing": (dict(lam_cf=0.0), dict()),
    "- KL to reference": (dict(beta_kl=0.0), dict()),
    "REINFORCE (no group norm.)": (dict(advantage="baseline"), dict()),
}


def stage_ablation(S):
    out = {}
    path = os.path.join(RES, "ablation.json")
    if os.path.exists(path):
        out = json.load(open(path))
    main = json.load(open(os.path.join(RES, "main.json")))
    out["Full (GRPO)"] = main["grpo"]["T+V+C"]
    out["SFT only"] = main["sft"]["T+V+C"]
    for name, (gkw, rkw) in ABLATIONS.items():
        if name in out:
            continue
        out[name] = []
        for seed in SEEDS:
            m = pickle.load(open(os.path.join(MOD, f"sft_s{seed}.pkl"), "rb"))
            m, _ = train_grpo(m, m.clone(), S["Btr"], S["Btr_cf"], S["cf_tr"], S["Bva"], S["F"],
                              GRPOConfig(epochs=GRPO_EPOCHS, seed=seed, **gkw), RewardConfig(**rkw), verbose=False)
            ev = evaluate_policy(m, S["F"], S["Bte"], S["Bte_cf"], S["cf_te"])
            out[name].append(strip(ev))
            print(f"{name} seed {seed}: F1 {ev['overall']['macro_f1']:.4f}", flush=True)
        jdump(out, "ablation.json")


def stage_lowres():
    out = {}
    for frac in ((0.0, 1.0) if os.environ.get("SMOKE") else (0.0, 0.1, 0.25, 0.5, 1.0)):
        rng = np.random.default_rng(0)

        def sub(tr, frac=frac):
            return [r for r in tr if r["lang"] != "hi" or rng.random() < frac]
        S = setup(sub)
        n_hi = sum(r["lang"] == "hi" for r in S["tr"])
        te_hi = [i for i, r in enumerate(S["te"]) if r["lang"] == "hi"]
        res = {"n_hi_train": n_hi}
        p, _ = tfidf_lr(S["tr"], S["te"], analyzer="char_wb", ngram=(2, 5))
        res["char_lr"] = {"hi": evaluate_probs([S["te"][i] for i in te_hi], S["Bte"].y[te_hi], p[te_hi])["overall"],
                          "all": evaluate_probs(S["te"], S["Bte"].y, p)["overall"]}
        m = new_policy(S, 0)
        train_sft(m, S["Btr"], S["Bva"], epochs=SFT_EPOCHS, verbose=False)
        p = m.predict_proba(S["Bte"])[:, 1]
        res["sft"] = {"hi": evaluate_probs([S["te"][i] for i in te_hi], S["Bte"].y[te_hi], p[te_hi])["overall"],
                      "all": evaluate_probs(S["te"], S["Bte"].y, p)["overall"]}
        m, _ = train_grpo(m, m.clone(), S["Btr"], S["Btr_cf"], S["cf_tr"], S["Bva"], S["F"],
                          GRPOConfig(epochs=GRPO_EPOCHS), RewardConfig(), verbose=False)
        p = m.predict_proba(S["Bte"])[:, 1]
        res["grpo"] = {"hi": evaluate_probs([S["te"][i] for i in te_hi], S["Bte"].y[te_hi], p[te_hi])["overall"],
                       "all": evaluate_probs(S["te"], S["Bte"].y, p)["overall"]}
        out[str(frac)] = res
        print(f"lowres frac={frac} n_hi={n_hi}: charLR {res['char_lr']['hi']['macro_f1']:.4f} "
              f"SFT {res['sft']['hi']['macro_f1']:.4f} GRPO {res['grpo']['hi']['macro_f1']:.4f}", flush=True)
        jdump(out, "lowres.json")


def stage_feedback():
    out = {}
    S = setup()
    hate_with_gold = None
    for frac in ((0.0, 0.3) if os.environ.get("SMOKE") else (0.0, 0.02, 0.05, 0.1, 0.3, 1.0)):
        rng = np.random.default_rng(1)
        S["Btr"].has_gold = np.array([rng.random() < frac for _ in range(len(S["Btr"]))])
        hate_with_gold = int((S["Btr"].has_gold & (S["Btr"].y == 1)).sum())
        res = {"n_rationales": hate_with_gold}
        m = new_policy(S, 0)
        train_sft(m, S["Btr"], S["Bva"], epochs=SFT_EPOCHS, verbose=False)
        ev = evaluate_policy(m, S["F"], S["Bte"], S["Bte_cf"], S["cf_te"])
        res["sft"] = {"plausibility": ev["plausibility"], "faithfulness": ev["faithfulness"], "overall": ev["overall"]}
        m, _ = train_grpo(m, m.clone(), S["Btr"], S["Btr_cf"], S["cf_tr"], S["Bva"], S["F"],
                          GRPOConfig(epochs=GRPO_EPOCHS), RewardConfig(), verbose=False)
        ev = evaluate_policy(m, S["F"], S["Bte"], S["Bte_cf"], S["cf_te"])
        res["grpo"] = {"plausibility": ev["plausibility"], "faithfulness": ev["faithfulness"], "overall": ev["overall"]}
        out[str(frac)] = res
        print(f"feedback frac={frac} ({hate_with_gold} rationales): SFT tokF1 {res['sft']['plausibility']['token_f1']:.3f} "
              f"comp {res['sft']['faithfulness']['comprehensiveness']:.3f} | GRPO tokF1 "
              f"{res['grpo']['plausibility']['token_f1']:.3f} comp {res['grpo']['faithfulness']['comprehensiveness']:.3f}",
              flush=True)
        jdump(out, "feedback.json")


def stage_xai(S, n_posts=10 if os.environ.get("SMOKE") else 300):
    from collections import defaultdict
    out = {}
    B, F, te = S["Bte"], S["F"], S["te"]
    gold_frac = float(np.mean([np.mean(r["rationale"]) for r in S["tr"] if r["label_true"] == 1]))
    for tag in ("grpo", "sft"):
        m = pickle.load(open(os.path.join(MOD, f"{tag}_s0.pkl"), "rb"))
        P = m.predict_proba(B)[:, 1]
        cand = [i for i in range(len(B)) if B.y[i] == 1 and P[i] >= 0.5 and len(B.tokens[i]) >= 3]
        rng = np.random.default_rng(0)
        sel = rng.choice(cand, size=min(n_posts, len(cand)), replace=False)
        res = {}
        explainers = dict(xai.EXPLAINERS)
        explainers["Random"] = lambda m_, F_, b_, i_: np.random.default_rng(int(i_)).random(len(b_.tokens[i_]))
        for name, fn in explainers.items():
            t0 = time.time()
            scores, golds, hards, comp, suff = [], [], [], [], []
            for i in sel:
                s = np.asarray(fn(m, F, B, i), dtype=float)
                scores.append(s)
                golds.append(B.gold[i])
                hards.append(topk_mask(s, gold_frac))
                c, su = xai.aopc_faithfulness(m, F, B, i, s)
                comp.append(c)
                suff.append(su)
            pl = plausibility(scores, golds, hards)
            res[name] = {"auprc": pl["auprc"], "token_f1": pl["token_f1"], "iou_f1": pl["iou_f1"],
                         "comprehensiveness": float(np.mean(comp)), "sufficiency": float(np.mean(suff)),
                         "sec_per_post": (time.time() - t0) / len(sel), "n": len(sel)}
            print(tag, name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
        out[tag] = res
        # modality Shapley by functionality (all test posts predicted hateful)
        idx = np.where(P >= 0.5)[0]
        phi, base = xai.modality_shapley(m, B, idx)
        by = defaultdict(list)
        for k, i in enumerate(idx):
            by[te[i]["func"]].append(phi[k])
        out[f"{tag}_modality_shapley"] = {f: np.mean(v, 0).tolist() for f, v in by.items()}
        out[f"{tag}_modality_shapley_abs_share"] = {
            f: (np.mean(np.abs(v), 0) / np.mean(np.abs(v), 0).sum()).tolist() for f, v in by.items()}
        jdump(out, "xai.json")
    # case studies
    m = pickle.load(open(os.path.join(MOD, "grpo_s0.pkl"), "rb"))
    cases = []
    want = [("MM_HATE", "en"), ("MM_BENIGN", "en"), ("CTX_HATE", "en"), ("CTX_BENIGN", "en"), ("N_COUNTER", "en"),
            ("H_THREAT", "hien"), ("H_DEROG", "hi"), ("H_IMPLICIT", "en")]
    for func, lang in want:
        for i, r in enumerate(te):
            if r["func"] == func and r["lang"] == lang and r["template_seen"]:
                text, info = xai.explain(m, F, B, i)
                cases.append({"id": r["id"], "func": func, "lang": lang, "text": r["text"], "parent": r["parent_text"],
                              "image_symbol": r["image_symbol"], "image_path": r["image_path"], "label": r["label"],
                              "tokens": r["tokens"], "ig": info["ig"].tolist(), "rationale": info["rationale"].tolist(),
                              "modality_phi": info["modality_phi"].tolist(), "p": float(info["p"]), "explanation": text})
                break
    out["cases"] = cases
    jdump(out, "xai.json")
    for c in cases:
        print("-" * 70)
        print(f"[{c['func']}/{c['lang']}] text: {c['text']} | parent: {c['parent']} | image: {c['image_symbol']} | y={c['label']}")
        print(c["explanation"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "main", "ablation", "lowres", "feedback", "xai"])
    a = ap.parse_args()
    if a.stage == "data":
        generate(GenConfig(out_dir=DATA))
    elif a.stage == "lowres":
        stage_lowres()
    elif a.stage == "feedback":
        stage_feedback()
    else:
        S = setup()
        {"main": stage_main, "ablation": stage_ablation, "xai": stage_xai}[a.stage](S)
