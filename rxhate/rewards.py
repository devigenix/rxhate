"""Reward functions for reward-optimised hate speech detection.

These are deliberately framework-agnostic (plain Python / NumPy) so that the *same*
reward drives (a) the NumPy fusion policy trained with GRPO in rxhate/grpo.py and
(b) an instruction-tuned LLM trained with TRL's GRPOTrainer in llm/train_grpo.py.

An action is a = (y_hat, z): a label and a token-level rationale mask.

    R(x, a) =  w_acc   * r_acc(y_hat, y)                 correctness (asymmetric costs)
             + w_fair  * r_fair(y_hat, y, identity)      penalise FP on benign identity mentions
             + w_faith * r_faith(p_full, p_suff, p_comp) rationale must be faithful
             + w_plaus * r_plaus(z, z*)                  agreement with human rationale (if any)
             + w_sparse* r_sparse(z)                     concise rationale
             + w_empty * r_empty(y_hat, z)               no rationale for non-hateful decisions
             + w_format* r_format(...)                   (LLM only) parseable output
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RewardConfig:
    w_acc: float = 1.0
    fn_cost: float = 1.0          # cost of missing hate
    fp_cost: float = 1.0          # cost of a false alarm
    w_fair: float = 0.5
    w_faith: float = 0.5
    w_plaus: float = 0.5
    w_sparse: float = 0.3
    rho: float = 0.6              # rationale budget (~ mean human-rationale length on train)
    w_empty: float = 0.3
    w_format: float = 0.5

    def to_dict(self):
        return asdict(self)


def r_acc(pred, gold, cfg):
    if pred == gold:
        return 1.0
    return -(cfg.fn_cost if gold == 1 else cfg.fp_cost)


def r_fair(pred, gold, has_identity):
    """Extra penalty for flagging a *non-hateful* post only because it mentions an identity
    group -- the dominant source of unintended bias in hate speech classifiers."""
    return -1.0 if (pred == 1 and gold == 0 and has_identity) else 0.0


def r_faith(pred, mask, p_full, p_suff, p_comp):
    """ERASER-style faithfulness for a hateful decision:
       comprehensiveness  p(hate|x) - p(hate|x \\ z)   (removing z should hurt)
       sufficiency gap    max(0, p(hate|x) - p(hate|z)) (z alone should suffice)"""
    if pred != 1:
        return 0.0
    if np.sum(mask) == 0:
        return -1.0
    return float((p_full - p_comp) - max(0.0, p_full - p_suff))


def r_plaus(pred, gold, mask, gold_mask):
    if gold_mask is None or pred != 1 or gold != 1:
        return 0.0
    m, g = np.asarray(mask, bool), np.asarray(gold_mask, bool)
    tp = (m & g).sum()
    if tp == 0:
        return 0.0
    p, r = tp / m.sum(), tp / g.sum()
    return float(2 * p * r / (p + r))


def r_sparse(pred, mask, rho):
    if pred != 1 or len(mask) == 0:
        return 0.0
    return -max(0.0, float(np.mean(mask)) - rho)


def r_empty(pred, mask):
    if pred != 0 or len(mask) == 0:
        return 0.0
    return -float(np.mean(mask))


def total_reward(pred, gold, mask, gold_mask, has_identity, p_full, p_suff, p_comp, cfg: RewardConfig):
    comps = {
        "acc": r_acc(pred, gold, cfg),
        "fair": r_fair(pred, gold, has_identity),
        "faith": r_faith(pred, mask, p_full, p_suff, p_comp),
        "plaus": r_plaus(pred, gold, mask, gold_mask),
        "sparse": r_sparse(pred, mask, cfg.rho),
        "empty": r_empty(pred, mask),
    }
    r = (cfg.w_acc * comps["acc"] + cfg.w_fair * comps["fair"] + cfg.w_faith * comps["faith"]
         + cfg.w_plaus * comps["plaus"] + cfg.w_sparse * comps["sparse"] + cfg.w_empty * comps["empty"])
    return r, comps


# ----------------------------------------------------------------------------------
# LLM completions:  {"label": "hateful" | "not_hateful", "target": str,
#                    "rationale": [str, ...], "explanation": str}
# ----------------------------------------------------------------------------------
LABELS = {"hateful": 1, "hate": 1, "1": 1, "not_hateful": 0, "non-hateful": 0, "not hateful": 0,
          "normal": 0, "0": 0}


def parse_completion(text):
    """Return dict(label, rationale, target, explanation, ok) from an LLM completion."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = LABELS.get(str(obj.get("label", "")).strip().lower())
            rat = obj.get("rationale", []) or []
            if isinstance(rat, str):
                rat = rat.split()
            return {"label": lab, "rationale": [str(w).lower() for w in rat], "target": obj.get("target"),
                    "explanation": obj.get("explanation", ""), "ok": lab is not None}
        except (json.JSONDecodeError, AttributeError):
            pass
    low = text.lower()
    lab = 0 if "not_hateful" in low or "not hateful" in low else (1 if "hateful" in low else None)
    return {"label": lab, "rationale": [], "target": None, "explanation": "", "ok": False}


def rationale_words_to_mask(tokens, words):
    ws = set(w.lower().strip(".,!?") for w in words)
    return np.array([1 if t.lower() in ws else 0 for t in tokens])


def r_format(parsed):
    return 1.0 if parsed["ok"] else -1.0
