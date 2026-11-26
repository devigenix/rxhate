"""TRL-compatible reward functions wrapping rxhate.rewards.

GRPOTrainer calls each function as  f(prompts=..., completions=..., **dataset_columns)
and expects a list[float].  Completions are strings (standard format) or lists of
{"role", "content"} messages (conversational format); both are handled.
"""
from __future__ import annotations

import pickle

import numpy as np

from rxhate.rewards import (RewardConfig, parse_completion, r_acc, r_faith, r_fair, r_format, r_plaus, r_sparse,
                            rationale_words_to_mask)

CFG = RewardConfig()


def _text(c):
    if isinstance(c, list):
        return c[-1]["content"] if c else ""
    return c


class FaithfulnessScorer:
    """Frozen judge p(hate | text) used to score LLM rationales for sufficiency and
    comprehensiveness without re-running the LLM.  Default: char n-gram LR trained by
    llm/train_grpo.py --build-scorer (any sklearn-style predict_proba model works)."""

    def __init__(self, path):
        self.vec, self.clf = pickle.load(open(path, "rb"))

    def __call__(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


SCORER = None


def set_scorer(path):
    global SCORER
    SCORER = FaithfulnessScorer(path)


def format_reward(completions, **kw):
    return [CFG.w_format * r_format(parse_completion(_text(c))) for c in completions]


def correctness_reward(completions, label, **kw):
    out = []
    for c, y in zip(completions, label):
        p = parse_completion(_text(c))
        out.append(CFG.w_acc * (r_acc(p["label"], int(y), CFG) if p["label"] is not None else -1.0))
    return out


def fairness_reward(completions, label, has_identity, **kw):
    return [CFG.w_fair * r_fair(parse_completion(_text(c))["label"], int(y), bool(h))
            for c, y, h in zip(completions, label, has_identity)]


def plausibility_reward(completions, label, tokens, rationale, has_gold_rationale, **kw):
    out = []
    for c, y, toks, gold, hg in zip(completions, label, tokens, rationale, has_gold_rationale):
        p = parse_completion(_text(c))
        if not hg or p["label"] is None:
            out.append(0.0)
            continue
        m = rationale_words_to_mask(toks, p["rationale"])
        out.append(CFG.w_plaus * r_plaus(p["label"], int(y), m, np.array(gold)))
    return out


def rationale_shape_reward(completions, tokens, **kw):
    """Sparsity for hateful answers, empty rationale for non-hateful answers, and a penalty
    for 'rationale' words that do not occur in the post (hallucinated evidence)."""
    out = []
    for c, toks in zip(completions, tokens):
        p = parse_completion(_text(c))
        m = rationale_words_to_mask(toks, p["rationale"])
        r = 0.0
        if p["label"] == 1:
            r += CFG.w_sparse * r_sparse(1, m, CFG.rho)
        elif p["label"] == 0 and p["rationale"]:
            r -= CFG.w_empty
        lower = set(t.lower() for t in toks)
        if p["rationale"]:
            r -= 0.5 * np.mean([w not in lower for w in p["rationale"]])
        out.append(r)
    return out


def faithfulness_reward(completions, tokens, **kw):
    if SCORER is None:
        return [0.0] * len(completions)
    full, suff, comp, idx = [], [], [], []
    parsed = [parse_completion(_text(c)) for c in completions]
    for k, (p, toks) in enumerate(zip(parsed, tokens)):
        if p["label"] == 1:
            m = rationale_words_to_mask(toks, p["rationale"]).astype(bool)
            full.append(" ".join(toks))
            suff.append(" ".join(t for t, z in zip(toks, m) if z))
            comp.append(" ".join(t for t, z in zip(toks, m) if not z))
            idx.append((k, m))
    out = [0.0] * len(completions)
    if idx:
        pf, ps, pc = SCORER(full), SCORER(suff), SCORER(comp)
        for j, (k, m) in enumerate(idx):
            out[k] = CFG.w_faith * r_faith(1, m, pf[j], ps[j], pc[j])
    return out


ALL_REWARDS = [format_reward, correctness_reward, fairness_reward, plausibility_reward, rationale_shape_reward,
               faithfulness_reward]
