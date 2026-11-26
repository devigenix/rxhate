"""Multimodal fusion policy  pi_theta(y, z | x)  implemented in NumPy.

    h_T = ReLU(W_T x_T + b_T)                  text encoder   (hashed n-grams, sparse)
    h_V = ReLU(W_V x_V + b_V)                  visual encoder (image descriptor)
    h_C = ReLU(W_C x_C + W_M m + b_C)          context encoder (parent post + metadata)
    h   = ReLU(W_F [h_T; h_V; h_C] + b_F)      late-fusion block
    p(y | x)   = softmax(W_O h + b_O)          label head  (the "decision")
    p(z_t | x) = sigmoid(w_r . r_t + v_r . h_T + b_r)   rationale head (the "explanation")

The pair (y, z) is the *action* of the policy: a label plus a token-level rationale,
factorised autoregressively as pi(y, z | x) = pi(y | x) * pi(z | x, y), where a rationale is
only produced for a hateful decision (pi(z = {} | x, y = 0) = 1).
This mirrors an LLM that generates "label + highlighted evidence" and lets the same
reward functions drive both the small model here and the LLM in llm/train_grpo.py.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp

from .metrics import classification_report


def relu(x):
    return np.maximum(x, 0.0)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        self.t += 1
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (gn + 1e-12))
        for k, g in grads.items():
            g = g * scale
            if self.wd and k.startswith("W"):
                g = g + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return gn


class FusionPolicy:
    MODALITIES = ("text", "image", "context")

    def __init__(self, d_text, d_img, d_ctx, d_meta, d_rat, h_text=128, h_img=48, h_ctx=48, h_fuse=64,
                 seed=0, modalities=("text", "image", "context"), rat_context=False):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.modalities = tuple(modalities)
        self.rat_context = rat_context  # condition rationale scores on the sentence encoding h_T

        def he(i, o, fan=None):
            return rng.normal(0, np.sqrt(2.0 / (fan or i)), (i, o))

        hf_in = h_text + h_img + h_ctx
        self.p = {
            "WT": he(d_text, h_text, fan=2), "bT": np.zeros(h_text),
            "WV": he(d_img, h_img), "bV": np.zeros(h_img),
            "WC": he(d_ctx, h_ctx, fan=2), "WM": he(d_meta, h_ctx), "bC": np.zeros(h_ctx),
            "WF": he(hf_in, h_fuse), "bF": np.zeros(h_fuse),
            "WO": he(h_fuse, 2) * 0.1, "bO": np.zeros(2),
            "wr": np.zeros(d_rat), "vr": np.zeros(h_text), "br": np.array([-1.0]),
        }
        self.dims = dict(h_text=h_text, h_img=h_img, h_ctx=h_ctx)

    # ------------------------------------------------------------------ helpers
    def clone(self):
        return copy.deepcopy(self)

    def modality_mask(self, n, drop=0.0, keep=None):
        m = np.ones((n, 3))
        for j, name in enumerate(self.MODALITIES):
            if name not in self.modalities:
                m[:, j] = 0.0
        if keep is not None:
            m = m * np.asarray(keep, dtype=float)
        if drop > 0:
            d = self.rng.random((n, 3)) < drop
            d[:, 0] &= self.rng.random(n) < 0.5  # drop text less often
            m = m * (~d)
        return m

    # ------------------------------------------------------------------ forward / backward
    def forward(self, XT, XV, XC, XM, mod=None, dropout=0.0):
        p = self.p
        n = XT.shape[0]
        if mod is None:
            mod = self.modality_mask(n)
        zT = np.asarray(XT @ p["WT"])
        zV = XV @ p["WV"]
        zC = np.asarray(XC @ p["WC"]) + XM @ p["WM"]
        aT = mod[:, :1] * zT + p["bT"]
        aV = mod[:, 1:2] * zV + p["bV"]
        aC = mod[:, 2:3] * zC + p["bC"]
        hT, hV, hC = relu(aT), relu(aV), relu(aC)
        u = np.concatenate([hT, hV, hC], 1)
        aF = u @ p["WF"] + p["bF"]
        hF = relu(aF)
        dmask = None
        if dropout > 0:
            dmask = (self.rng.random(hF.shape) >= dropout) / (1 - dropout)
            hF = hF * dmask
        logits = hF @ p["WO"] + p["bO"]
        cache = dict(XT=XT, XV=XV, XC=XC, XM=XM, mod=mod, aT=aT, aV=aV, aC=aC, u=u, aF=aF, hF=hF,
                     dmask=dmask, hT=hT)
        return logits, cache

    def backward(self, cache, dlogits, need_input_grad_text=False):
        p = self.p
        g = {}
        g["WO"] = cache["hF"].T @ dlogits
        g["bO"] = dlogits.sum(0)
        dhF = dlogits @ p["WO"].T
        if cache["dmask"] is not None:
            dhF = dhF * cache["dmask"]
        daF = dhF * (cache["aF"] > 0)
        g["WF"] = cache["u"].T @ daF
        g["bF"] = daF.sum(0)
        du = daF @ p["WF"].T
        hT_, hV_ = self.dims["h_text"], self.dims["h_img"]
        daT = du[:, :hT_] * (cache["aT"] > 0)
        daV = du[:, hT_:hT_ + hV_] * (cache["aV"] > 0)
        daC = du[:, hT_ + hV_:] * (cache["aC"] > 0)
        mod = cache["mod"]
        g["bT"], g["bV"], g["bC"] = daT.sum(0), daV.sum(0), daC.sum(0)
        g["WT"] = np.asarray(cache["XT"].T @ (daT * mod[:, :1]))
        g["WV"] = cache["XV"].T @ (daV * mod[:, 1:2])
        dC = daC * mod[:, 2:3]
        g["WC"] = np.asarray(cache["XC"].T @ dC)
        g["WM"] = cache["XM"].T @ dC
        if need_input_grad_text:
            return g, (daT * mod[:, :1]) @ p["WT"].T  # dense N x D_T (use on small batches only)
        return g

    def predict_proba(self, batch, mod=None, bs=1024, XT=None):
        XT = batch.XT if XT is None else XT
        out = []
        for s in range(0, XT.shape[0], bs):
            m = None if mod is None else mod[s:s + bs]
            lg, _ = self.forward(XT[s:s + bs], batch.XV[s:s + bs], batch.XC[s:s + bs], batch.XM[s:s + bs], mod=m)
            out.append(softmax(lg))
        return np.concatenate(out)

    # ------------------------------------------------------------------ rationale head
    def rationale_scores(self, R_list, hT):
        """R_list: list of (n_i x D_R) csr; hT: (B x h_text). Returns list of score arrays."""
        if not R_list:
            return []
        Rs = sp.vstack(R_list).tocsr()
        lens = [r.shape[0] for r in R_list]
        owner = np.repeat(np.arange(len(R_list)), lens)
        s = Rs @ self.p["wr"] + self.p["br"][0]
        if self.rat_context:
            s = s + hT[owner] @ self.p["vr"]
        out, k = [], 0
        for L in lens:
            out.append(s[k:k + L])
            k += L
        return out, (Rs, owner, lens)

    def rationale_backward(self, rcache, hT, ds_list):
        Rs, owner, _ = rcache
        ds = np.concatenate(ds_list)
        g = {"wr": np.asarray(Rs.T @ ds).ravel(), "br": np.array([ds.sum()])}
        g["vr"] = hT[owner].T @ ds if self.rat_context else np.zeros_like(self.p["vr"])
        return g

    def text_hidden(self, batch, bs=1024):
        hs = []
        for s in range(0, batch.XT.shape[0], bs):
            hs.append(relu(np.asarray(batch.XT[s:s + bs] @ self.p["WT"]) + self.p["bT"]))
        return np.concatenate(hs)

    def rationale_probs(self, batch):
        hT = self.text_hidden(batch)
        s, _ = self.rationale_scores(batch.R, hT)
        return [sigmoid(x) for x in s]


# ----------------------------------------------------------------------------------
# Supervised fine-tuning (stage 1)
# ----------------------------------------------------------------------------------

def train_sft(model, train, val, epochs=12, bs=64, lr=2e-3, mod_drop=0.15, dropout=0.2, lam_rat=1.0,
              wd=1e-5, patience=3, verbose=True, log=None):
    opt = Adam(model.p, lr=lr, wd=wd)
    n = len(train)
    best, best_p, bad = -1, None, 0
    for ep in range(epochs):
        perm = model.rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            b = train.subset(idx)
            mod = model.modality_mask(len(idx), drop=mod_drop)
            logits, cache = model.forward(b.XT, b.XV, b.XC, b.XM, mod=mod, dropout=dropout)
            P = softmax(logits)
            Y = np.eye(2)[b.y]
            loss = -np.log(P[np.arange(len(idx)), b.y] + 1e-12).mean()
            grads = model.backward(cache, (P - Y) / len(idx))
            # rationale head  pi(z | x, y=1): BCE on hateful posts that carry human rationales
            gi = np.where(b.has_gold & (b.y == 1))[0]
            if len(gi) and lam_rat > 0:
                hT = cache["hT"][gi]
                scores, rc = model.rationale_scores([b.R[i] for i in gi], hT)
                ntok = sum(len(x) for x in scores)
                ds = [lam_rat * (sigmoid(sc) - b.gold[i]) / ntok for sc, i in zip(scores, gi)]
                grads.update(model.rationale_backward(rc, hT, ds))
            opt.step(model.p, grads)
            tot += loss * len(idx)
        P = model.predict_proba(val)
        rep = classification_report(val.y, P[:, 1])
        if verbose:
            print(f"  [sft] epoch {ep + 1:2d} loss {tot / n:.4f}  val macroF1 {rep['macro_f1']:.4f}")
        if log is not None:
            log.append({"epoch": ep + 1, "loss": tot / n, "val_macro_f1": rep["macro_f1"]})
        if rep["macro_f1"] > best + 1e-4:
            best, best_p, bad = rep["macro_f1"], copy.deepcopy(model.p), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.p = best_p
    return model
