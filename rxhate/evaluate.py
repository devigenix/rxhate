"""One-stop evaluation of a trained policy."""
from __future__ import annotations

import numpy as np

from .grpo import faithfulness_probs, greedy_actions
from .metrics import (bias_metrics, classification_report, counterfactual_fairness, per_slice, plausibility)


def evaluate_probs(records, y, p1, p1_cf=None, cf_mask=None):
    out = {"overall": classification_report(y, p1)}
    for key in ("lang", "func", "template_seen", "target_category"):
        out[f"by_{key}"] = {str(k): v for k, v in per_slice(records, y, p1, key).items()}
    b = bias_metrics(records, y, p1)
    out["bias"] = {k: v for k, v in b.items() if k != "per_group"}
    out["bias_per_group"] = b["per_group"]
    if p1_cf is not None:
        out["counterfactual"] = counterfactual_fairness(p1, p1_cf, cf_mask)
    return out


def evaluate_policy(model, featurizer, B, Bcf, cf_mask):
    P, yhat, rp, masks = greedy_actions(model, B)
    p1 = P[:, 1]
    p1_cf = model.predict_proba(Bcf)[:, 1]
    out = evaluate_probs(B.records, B.y, p1, p1_cf, cf_mask)
    # explanation quality of the policy's own rationale on gold-hateful posts
    hate = [i for i in range(len(B)) if B.y[i] == 1]
    hard = [rp[i] >= 0.5 if rp[i].max() >= 0.5 else (rp[i] == rp[i].max()) for i in range(len(B))]
    out["plausibility"] = plausibility([rp[i] for i in hate], [B.gold[i] for i in hate], [hard[i] for i in hate])
    tp = [i for i in hate if yhat[i] == 1]
    items = [(i, hard[i]) for i in tp]
    ps = faithfulness_probs(model, featurizer, B, items)
    pc = faithfulness_probs(model, featurizer, B, [(i, ~k) for i, k in items])
    out["faithfulness"] = {
        "comprehensiveness": float(np.mean(p1[tp] - pc)),
        "sufficiency": float(np.mean(p1[tp] - ps)),
        "rationale_len_frac": float(np.mean([hard[i].mean() for i in tp])),
        "gold_len_frac": float(np.mean([B.gold[i].mean() for i in tp])),
    }
    out["_p1"] = p1.tolist()
    return out
