"""Stage 2: reward optimisation with Group Relative Policy Optimisation (GRPO).

For every post we sample a group of G actions a_g = (y_hat_g, z_g) from the current
policy, score each with the reward model of rxhate/rewards.py, and use the group-
normalised reward as the advantage (Shao et al., 2024):

    A_g = (R_g - mean(R_1..G)) / (std(R_1..G) + eps)

The policy is updated with the PPO clipped surrogate (Schulman et al., 2017), an exact
KL penalty to the frozen SFT reference policy, and counterfactual logit pairing
(Garg et al., 2019) that ties p(y|x) to p(y|x with the identity term swapped):

  L = -E[min(rho A, clip(rho, 1-eps, 1+eps) A)] + beta KL(pi || pi_ref) + lam_cf (l(x) - l(x_cf))^2
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

from .metrics import classification_report
from .model import Adam, sigmoid, softmax
from .rewards import RewardConfig, total_reward


@dataclass
class GRPOConfig:
    epochs: int = 4
    batch_size: int = 32
    group_size: int = 8
    inner_steps: int = 2
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    lam_cf: float = 0.5
    lr: float = 5e-4
    advantage: str = "group"     # "group" (GRPO) | "baseline" (REINFORCE with moving-average baseline)
    use_faith: bool = True
    use_plaus: bool = True
    use_fair: bool = True
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _logits_diff(logits):
    return logits[:, 1] - logits[:, 0]


def faithfulness_probs(model, featurizer, b, items):
    """items: list of (row_in_batch, keep_mask). Returns p(hate) for the restricted texts."""
    if not items:
        return np.zeros(0)
    rows, cols, vals = [], [], []
    for r, (i, keep) in enumerate(items):
        idx, val = featurizer.text.featurize_tokens(b.tokens[i], keep)
        rows.append(np.full(len(idx), r))
        cols.append(idx)
        vals.append(val)
    XT = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(items), featurizer.text.dim))
    src = np.array([i for i, _ in items])
    logits, _ = model.forward(XT, b.XV[src], b.XC[src], b.XM[src])
    return softmax(logits)[:, 1]


def score_actions(model, featurizer, b, P, actions, masks, rcfg, cfg):
    """actions: (G,B) ints; masks: list over g of list over i of bool arrays."""
    G, B = actions.shape
    suff_items, comp_items, where = [], [], []
    for g in range(G):
        for i in range(B):
            if actions[g, i] == 1 and masks[g][i].any() and cfg.use_faith:
                suff_items.append((i, masks[g][i]))
                comp_items.append((i, ~masks[g][i]))
                where.append((g, i))
    ps = faithfulness_probs(model, featurizer, b, suff_items)
    pc = faithfulness_probs(model, featurizer, b, comp_items)
    pmap = {w: (ps[k], pc[k]) for k, w in enumerate(where)}
    R = np.zeros((G, B))
    comps_sum = {}
    for g in range(G):
        for i in range(B):
            p_full = P[i, 1]
            p_s, p_c = pmap.get((g, i), (p_full, p_full))
            gm = b.gold[i] if (b.has_gold[i] and cfg.use_plaus) else None
            r, comps = total_reward(int(actions[g, i]), int(b.y[i]), masks[g][i], gm,
                                    bool(b.identity[i]) and cfg.use_fair, p_full, p_s, p_c, rcfg)
            if not cfg.use_faith and actions[g, i] == 1:
                r -= rcfg.w_faith * comps["faith"]  # remove faithfulness term entirely
            R[g, i] = r
            for k, v in comps.items():
                comps_sum[k] = comps_sum.get(k, 0.0) + v
    return R, {k: v / (G * B) for k, v in comps_sum.items()}


def greedy_actions(model, b):
    P = model.predict_proba(b)
    yhat = (P[:, 1] >= 0.5).astype(int)
    rp = model.rationale_probs(b)
    masks = []
    for i, pr in enumerate(rp):
        m = pr >= 0.5
        if yhat[i] == 1 and not m.any():
            m[np.argmax(pr)] = True
        if yhat[i] == 0:
            m = np.zeros_like(m)
        masks.append(m)
    return P, yhat, rp, masks


def evaluate_reward(model, featurizer, b, rcfg, cfg, bs=256):
    tot, n = 0.0, 0
    for s in range(0, len(b), bs):
        sb = b.subset(np.arange(s, min(len(b), s + bs)))
        P, yhat, _, masks = greedy_actions(model, sb)
        R, _ = score_actions(model, featurizer, sb, P, yhat[None, :], [masks], rcfg, cfg)
        tot += R.sum()
        n += len(sb)
    return tot / n


def train_grpo(policy, ref, train, train_cf, cf_avail, val, featurizer, cfg=GRPOConfig(), rcfg=RewardConfig(),
               verbose=True, log=None):
    rng = np.random.default_rng(cfg.seed)
    opt = Adam(policy.p, lr=cfg.lr)
    n = len(train)
    best_score, best_p = -np.inf, copy.deepcopy(policy.p)
    baseline = 0.0
    hist = []
    for ep in range(cfg.epochs):
        perm = rng.permutation(n)
        ep_r, ep_comps, steps = 0.0, {}, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            b = train.subset(idx)
            B, G = len(idx), cfg.group_size
            # ---------------- rollout with the current (old) policy
            logits, cache = policy.forward(b.XT, b.XV, b.XC, b.XM)
            P_old = softmax(logits)
            s_old, _ = policy.rationale_scores(b.R, cache["hT"])
            sig_old = [sigmoid(x) for x in s_old]
            actions = (rng.random((G, B)) < P_old[None, :, 1]).astype(int)
            # rationale only for hateful decisions: pi(z | x, y=0) = delta(empty)
            masks = [[(rng.random(len(sig_old[i])) < sig_old[i]) if actions[g, i] == 1
                      else np.zeros(len(sig_old[i]), bool) for i in range(B)] for g in range(G)]
            R, comps = score_actions(policy, featurizer, b, P_old, actions, masks, rcfg, cfg)
            if cfg.advantage == "group":
                A = (R - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + 1e-4)
            else:
                A = R - baseline
                baseline = 0.9 * baseline + 0.1 * R.mean()
            logp_old = np.log(P_old[np.arange(B)[None, :], actions] + 1e-12)
            for i in range(B):
                lb = np.log(sig_old[i] + 1e-12)
                lnb = np.log(1 - sig_old[i] + 1e-12)
                for g in range(G):
                    if actions[g, i] == 1:
                        logp_old[g, i] += np.where(masks[g][i], lb, lnb).sum()
            # reference policy (frozen SFT)
            lg_ref, cache_ref = ref.forward(b.XT, b.XV, b.XC, b.XM)
            P_ref = softmax(lg_ref)
            s_ref, _ = ref.rationale_scores(b.R, cache_ref["hT"])
            # counterfactual partners
            cf_rows = np.where(cf_avail[idx])[0]
            bcf = train_cf.subset(idx[cf_rows]) if len(cf_rows) else None

            for _ in range(cfg.inner_steps):
                logits, cache = policy.forward(b.XT, b.XV, b.XC, b.XM)
                P = softmax(logits)
                sc, rcache = policy.rationale_scores(b.R, cache["hT"])
                sig = [sigmoid(x) for x in sc]
                logp = np.log(P[np.arange(B)[None, :], actions] + 1e-12)
                for i in range(B):
                    lb, lnb = np.log(sig[i] + 1e-12), np.log(1 - sig[i] + 1e-12)
                    for g in range(G):
                        if actions[g, i] == 1:
                            logp[g, i] += np.where(masks[g][i], lb, lnb).sum()
                ratio = np.exp(np.clip(logp - logp_old, -20, 20))
                active = ((A >= 0) & (ratio <= 1 + cfg.clip_eps)) | ((A < 0) & (ratio >= 1 - cfg.clip_eps))
                c = np.where(active, -A * ratio, 0.0) / (B * G)          # dL/dlogpi
                onehot = np.stack([1 - actions, actions], -1)            # (G,B,2)
                dlogits = (c[..., None] * (onehot - P[None])).sum(0)     # (B,2)
                # exact KL(pi || pi_ref) on the label distribution
                logratio = np.log(P + 1e-12) - np.log(P_ref + 1e-12)
                kl = (P * logratio).sum(1, keepdims=True)
                dlogits += cfg.beta_kl * P * (logratio - kl) / B
                ds = []
                for i in range(B):
                    zi = np.stack([masks[g][i] for g in range(G)]).astype(float)   # (G, n_i)
                    on = actions[:, i:i + 1]                                         # rationale exists only if y=1
                    d = (on * c[:, i:i + 1] * (zi - sig[i][None])).sum(0)
                    d += cfg.beta_kl * sig[i] * (1 - sig[i]) * (sc[i] - s_ref[i]) / (B * len(sig[i]))
                    ds.append(d)
                # counterfactual logit pairing
                g_cf = None
                if bcf is not None and cfg.lam_cf > 0:
                    lg_cf, cache_cf = policy.forward(bcf.XT, bcf.XV, bcf.XC, bcf.XM)
                    dd = _logits_diff(logits[cf_rows]) - _logits_diff(lg_cf)
                    coef = 2 * cfg.lam_cf * dd / max(1, len(cf_rows))
                    dlogits[cf_rows, 1] += coef
                    dlogits[cf_rows, 0] -= coef
                    g_cf = policy.backward(cache_cf, np.stack([coef, -coef], 1))
                grads = policy.backward(cache, dlogits)
                if g_cf is not None:
                    for k in g_cf:
                        grads[k] = grads[k] + g_cf[k]
                grads.update(policy.rationale_backward(rcache, cache["hT"], ds))
                opt.step(policy.p, grads)
            ep_r += R.mean()
            steps += 1
            for k, v in comps.items():
                ep_comps[k] = ep_comps.get(k, 0.0) + v
        val_reward = evaluate_reward(policy, featurizer, val, rcfg, cfg)
        val_f1 = classification_report(val.y, policy.predict_proba(val)[:, 1])["macro_f1"]
        rec = {"epoch": ep + 1, "train_reward": ep_r / steps, "val_reward": val_reward, "val_macro_f1": val_f1,
               **{f"r_{k}": v / steps for k, v in ep_comps.items()}}
        hist.append(rec)
        if log is not None:
            log.append(rec)
        if verbose:
            print(f"  [grpo] epoch {ep + 1} train R {rec['train_reward']:.3f}  val R {val_reward:.3f}  val F1 {val_f1:.4f}  "
                  + " ".join(f"{k}={v / steps:.3f}" for k, v in ep_comps.items()))
        if val_reward > best_score:
            best_score, best_p = val_reward, copy.deepcopy(policy.p)
    policy.p = best_p
    return policy, hist
