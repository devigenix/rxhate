"""Evaluation: classification, fairness/bias, and explanation (plausibility + faithfulness)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_report(y, p1, thr=0.5):
    y = np.asarray(y)
    yhat = (np.asarray(p1) >= thr).astype(int)
    out = {
        "acc": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "f1_hate": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
    }
    out["auroc"] = roc_auc_score(y, p1) if len(set(y.tolist())) > 1 else float("nan")
    return out


def per_slice(records, y, p1, key):
    vals = sorted(set(r[key] for r in records if r[key] is not None))
    out = {}
    for v in vals:
        idx = np.array([i for i, r in enumerate(records) if r[key] == v])
        yy, pp = np.asarray(y)[idx], np.asarray(p1)[idx]
        yhat = (pp >= 0.5).astype(int)
        out[v] = {"n": len(idx), "acc": float((yhat == yy).mean()),
                  "macro_f1": f1_score(yy, yhat, average="macro", zero_division=0)}
    return out


# ------------------------------------------------------------------ fairness
def _auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return np.nan
    return roc_auc_score(y, s)


def bias_metrics(records, y, p1, group_key="group", power=-5):
    """Subgroup AUC, BPSN AUC, BNSP AUC (Borkan et al., 2019), their generalised mean
    (as used in HateXplain), and FPR/FNR equality differences (Dixon et al., 2018)."""
    y = np.asarray(y)
    p1 = np.asarray(p1)
    yhat = (p1 >= 0.5).astype(int)
    groups = sorted(set(r[group_key] for r in records if r[group_key]))
    in_g = {g: np.array([r[group_key] == g for r in records]) for g in groups}
    fpr_all = ((yhat == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    fnr_all = ((yhat == 0) & (y == 1)).sum() / max(1, (y == 1).sum())
    per = {}
    for g in groups:
        m = in_g[g]
        sub = _auc(y[m], p1[m])
        bpsn_m = (m & (y == 0)) | (~m & (y == 1))
        bnsp_m = (m & (y == 1)) | (~m & (y == 0))
        neg, pos = m & (y == 0), m & (y == 1)
        per[g] = {
            "n": int(m.sum()),
            "subgroup_auc": sub,
            "bpsn_auc": _auc(y[bpsn_m], p1[bpsn_m]),
            "bnsp_auc": _auc(y[bnsp_m], p1[bnsp_m]),
            "fpr": float(yhat[neg].mean()) if neg.any() else np.nan,
            "fnr": float(1 - yhat[pos].mean()) if pos.any() else np.nan,
        }

    def gmean(key):
        v = np.array([per[g][key] for g in groups if not np.isnan(per[g][key])])
        return float(np.power(np.mean(np.power(v, power)), 1.0 / power)) if len(v) else np.nan

    fped = float(np.nansum([abs(fpr_all - per[g]["fpr"]) for g in groups]))
    fned = float(np.nansum([abs(fnr_all - per[g]["fnr"]) for g in groups]))
    # false-positive rate on non-hateful posts that mention an identity (neutral/positive/counter-speech)
    ident_neg = np.array([bool(r[group_key]) for r in records]) & (y == 0)
    return {
        "gmb_subgroup_auc": gmean("subgroup_auc"),
        "gmb_bpsn_auc": gmean("bpsn_auc"),
        "gmb_bnsp_auc": gmean("bnsp_auc"),
        "fped": fped,
        "fned": fned,
        "identity_fpr": float(yhat[ident_neg].mean()) if ident_neg.any() else np.nan,
        "per_group": per,
    }


def counterfactual_fairness(p1, p1_cf, mask):
    """Counterfactual token fairness (Garg et al., 2019): swap the identity term."""
    p1, p1_cf, mask = np.asarray(p1), np.asarray(p1_cf), np.asarray(mask, bool)
    d = np.abs(p1[mask] - p1_cf[mask])
    flips = ((p1[mask] >= 0.5) != (p1_cf[mask] >= 0.5)).mean()
    return {"ctf_gap": float(d.mean()), "cf_flip_rate": float(flips)}


# ------------------------------------------------------------------ explanations
def token_f1(pred_mask, gold_mask):
    pred_mask, gold_mask = np.asarray(pred_mask, bool), np.asarray(gold_mask, bool)
    tp = (pred_mask & gold_mask).sum()
    if pred_mask.sum() == 0 and gold_mask.sum() == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred_mask.sum(), tp / gold_mask.sum()
    return 2 * p * r / (p + r)


def iou(pred_mask, gold_mask):
    a, b = np.asarray(pred_mask, bool), np.asarray(gold_mask, bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def plausibility(scores_list, gold_list, hard_list):
    """Token-level agreement with human rationales (ERASER / HateXplain protocol)."""
    auprc, f1s, ious = [], [], []
    for s, g, h in zip(scores_list, gold_list, hard_list):
        g = np.asarray(g)
        if g.sum() == 0 or g.sum() == len(g):
            continue
        auprc.append(average_precision_score(g, s))
        f1s.append(token_f1(h, g))
        ious.append(iou(h, g))
    ious = np.array(ious)
    return {"auprc": float(np.mean(auprc)), "token_f1": float(np.mean(f1s)),
            "iou_f1": float((ious >= 0.5).mean()), "n": len(auprc)}


def topk_mask(scores, frac):
    n = len(scores)
    k = max(1, int(np.ceil(frac * n)))
    m = np.zeros(n, bool)
    m[np.argsort(-np.asarray(scores), kind="stable")[:k]] = True
    return m
