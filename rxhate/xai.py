"""Explainable-AI toolkit for the fusion policy.

Token level (text modality):
  * integrated_gradients  -- Sundararajan et al. (2017), zero baseline, Riemann sum
  * occlusion             -- leave-one-token-out
  * lime                  -- Ribeiro et al. (2016), weighted ridge surrogate over token masks
  * kernel_shap           -- Lundberg & Lee (2017), Shapley-kernel weighted regression
  * learned rationale     -- the policy's own rationale head pi(z | x, y=1)
Modality level:
  * modality_shapley      -- exact Shapley values of {text, image, context} on the hate logit
Natural language:
  * explain               -- a human-readable report combining all of the above
"""
from __future__ import annotations

from itertools import combinations
from math import comb, factorial

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from .features import IDENTITY_TERMS
from .model import sigmoid, softmax
from .text import normalize_token


def _perturbed_text_matrix(featurizer, tokens, keeps):
    rows, cols, vals = [], [], []
    for r, keep in enumerate(keeps):
        idx, val = featurizer.text.featurize_tokens(tokens, keep)
        rows.append(np.full(len(idx), r))
        cols.append(idx)
        vals.append(val)
    if sum(len(c) for c in cols) == 0:
        return sp.csr_matrix((len(keeps), featurizer.text.dim))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(keeps), featurizer.text.dim))


def _hate_logit(model, XT, b, i, mod=None):
    n = XT.shape[0]
    rep = np.full(n, i)
    lg, cache = model.forward(XT, b.XV[rep], b.XC[rep], b.XM[rep], mod=mod)
    return lg[:, 1] - lg[:, 0], lg, cache


def predict_on_masks(model, featurizer, b, i, keeps, as_prob=True):
    XT = _perturbed_text_matrix(featurizer, b.tokens[i], keeps)
    d, lg, _ = _hate_logit(model, XT, b, i)
    return softmax(lg)[:, 1] if as_prob else d


# ------------------------------------------------------------------ token attributions
def integrated_gradients(model, featurizer, b, i, steps=32):
    x = b.XT[i]
    idx, val = x.indices, x.data
    alphas = (np.arange(steps) + 0.5) / steps
    XT = sp.vstack([x * a for a in alphas]).tocsr()
    rep = np.full(steps, i)
    lg, cache = model.forward(XT, b.XV[rep], b.XC[rep], b.XM[rep])
    dl = np.zeros((steps, 2))
    dl[:, 1], dl[:, 0] = 1.0, -1.0
    _, gin = model.backward(cache, dl, need_input_grad_text=True)
    feat_attr = gin[:, idx].mean(0) * val
    shares = featurizer.text.token_feature_shares(b.tokens[i])
    tok = np.zeros(len(b.tokens[i]))
    for j, a in zip(idx, feat_attr):
        for t, w in shares.get(int(j), []):
            tok[t] += a * w
    return tok


def occlusion(model, featurizer, b, i):
    n = len(b.tokens[i])
    keeps = [np.ones(n, bool)] + [~np.eye(n, dtype=bool)[t] for t in range(n)]
    p = predict_on_masks(model, featurizer, b, i, keeps, as_prob=False)
    return p[0] - p[1:]


def lime(model, featurizer, b, i, n_samples=300, width=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(b.tokens[i])
    keep_p = rng.uniform(0.0, 1.0, n_samples)
    Z = rng.random((n_samples, n)) < keep_p[:, None]
    Z[0] = True
    f = predict_on_masks(model, featurizer, b, i, list(Z))
    cos = Z.sum(1) / (np.sqrt(np.maximum(Z.sum(1), 1)) * np.sqrt(n))
    d = 1 - cos
    w = np.exp(-(d ** 2) / width ** 2)
    reg = Ridge(alpha=1.0).fit(Z.astype(float), f, sample_weight=w)
    return reg.coef_


def kernel_shap(model, featurizer, b, i, n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(b.tokens[i])
    if n == 1:
        return np.array([predict_on_masks(model, featurizer, b, i, [np.ones(1, bool)], False)[0]
                         - predict_on_masks(model, featurizer, b, i, [np.zeros(1, bool)], False)[0]])
    sizes = np.arange(1, n)
    kw = np.array([(n - 1) / (comb(n, s) * s * (n - s)) for s in sizes]) * np.array([comb(n, s) for s in sizes])
    kw /= kw.sum()
    Z = [np.zeros(n, bool), np.ones(n, bool)]
    W = [1e6, 1e6]
    for _ in range(n_samples):
        s = rng.choice(sizes, p=kw)
        z = np.zeros(n, bool)
        z[rng.choice(n, s, replace=False)] = True
        Z.append(z)
        W.append(1.0)
    Z = np.array(Z)
    f = predict_on_masks(model, featurizer, b, i, list(Z), as_prob=False)
    X = np.hstack([np.ones((len(Z), 1)), Z.astype(float)])
    sw = np.sqrt(np.array(W))[:, None]
    coef, *_ = np.linalg.lstsq(X * sw, f * sw[:, 0], rcond=None)
    return coef[1:]


def learned_rationale(model, b, i):
    hT = model.text_hidden(b.subset([i]))
    s, _ = model.rationale_scores([b.R[i]], hT)
    return sigmoid(s[0])


EXPLAINERS = {
    "Integrated Gradients": lambda m, F, b, i: integrated_gradients(m, F, b, i),
    "Occlusion": lambda m, F, b, i: occlusion(m, F, b, i),
    "LIME": lambda m, F, b, i: lime(m, F, b, i),
    "KernelSHAP": lambda m, F, b, i: kernel_shap(m, F, b, i),
    "Learned rationale (ours)": lambda m, F, b, i: learned_rationale(m, b, i),
}


# ------------------------------------------------------------------ faithfulness (ERASER)
def aopc_faithfulness(model, featurizer, b, i, scores, bins=(0.1, 0.2, 0.3, 0.5)):
    """Comprehensiveness and sufficiency averaged over top-k% bins (DeYoung et al., 2020)."""
    n = len(scores)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keeps_comp, keeps_suff = [np.ones(n, bool)], []
    for frac in bins:
        k = max(1, int(np.ceil(frac * n)))
        m = np.zeros(n, bool)
        m[order[:k]] = True
        keeps_comp.append(~m)
        keeps_suff.append(m)
    p = predict_on_masks(model, featurizer, b, i, keeps_comp + keeps_suff)
    full = p[0]
    comp = full - p[1:1 + len(bins)]
    suff = full - p[1 + len(bins):]
    return float(comp.mean()), float(suff.mean())


# ------------------------------------------------------------------ modality attribution
def modality_shapley(model, b, idx):
    """Exact Shapley values for players {text, image, context}; value = hate logit with the
    absent modalities' inputs zeroed (the model was trained with modality dropout, so this
    'absent' state is in-distribution)."""
    sub = b.subset(idx)
    players = 3
    vals = {}
    for r in range(players + 1):
        for S in combinations(range(players), r):
            keep = np.zeros(3)
            keep[list(S)] = 1
            mod = np.tile(keep, (len(idx), 1))
            lg, _ = model.forward(sub.XT, sub.XV, sub.XC, sub.XM, mod=mod)
            vals[S] = lg[:, 1] - lg[:, 0]
    phi = np.zeros((len(idx), players))
    for j in range(players):
        others = [k for k in range(players) if k != j]
        for r in range(players):
            for S in combinations(others, r):
                w = factorial(r) * factorial(players - r - 1) / factorial(players)
                Sj = tuple(sorted(S + (j,)))
                phi[:, j] += w * (vals[Sj] - vals[S])
    return phi, vals[()]


# ------------------------------------------------------------------ natural-language explanation
def explain(model, featurizer, b, i, top_k=4):
    p = model.predict_proba(b.subset([i]))[0, 1]
    label = "HATEFUL" if p >= 0.5 else "NOT HATEFUL"
    toks = b.tokens[i]
    ig = integrated_gradients(model, featurizer, b, i)
    rat = learned_rationale(model, b, i)
    phi, base = modality_shapley(model, b, [i])
    phi = phi[0]
    share = np.abs(phi) / (np.abs(phi).sum() + 1e-9)
    targets = [t for t in toks if normalize_token(t) in IDENTITY_TERMS]
    order = np.argsort(-ig)[:top_k]
    lines = [f"Prediction: {label}  (p_hate = {p:.2f})"]
    if p >= 0.5:
        ev = [toks[t] for t in range(len(toks)) if rat[t] >= 0.5] or [toks[int(np.argmax(rat))]]
        lines.append(f"Evidence (model rationale): {' '.join(ev)}")
    lines.append("Most influential words (IG): " + ", ".join(f"{toks[t]} ({ig[t]:+.2f})" for t in order))
    lines.append("Modality contributions to the hate logit: " + ", ".join(
        f"{n} {phi[k]:+.2f} ({share[k]:.0%})" for k, n in enumerate(["text", "image", "context"])))
    if targets:
        lines.append(f"Targeted identity mention(s): {', '.join(targets)}")
    dom = ["text", "image", "context"][int(np.argmax(np.abs(phi)))]
    if dom != "text":
        lines.append(f"Note: the decision is driven mainly by the {dom}, not by the words alone.")
    return "\n".join(lines), {"p": p, "ig": ig, "rationale": rat, "modality_phi": phi}
