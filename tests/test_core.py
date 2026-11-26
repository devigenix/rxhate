import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rxhate import rewards as R  # noqa: E402
from rxhate.data import loaders  # noqa: E402
from rxhate.data.synthetic import GenConfig, parse_rationale, render_text  # noqa: E402
from rxhate.features import Batch  # noqa: E402
from rxhate.metrics import bias_metrics, token_f1  # noqa: E402
from rxhate.model import FusionPolicy, softmax  # noqa: E402
from rxhate.text import HashedFeaturizer, normalize_token  # noqa: E402
from rxhate.xai import modality_shapley  # noqa: E402


# ------------------------------------------------------------------ text
def test_normalisation_handles_leet_and_variants():
    assert normalize_token("P@r4sites") == "parasites"
    assert normalize_token("nhi") == "nahi"
    assert normalize_token("sooooo") == "soo"


def test_feature_subsets_and_shares():
    f = HashedFeaturizer(dim=2 ** 12)
    toks = ["they", "are", "vermin"]
    idx, val = f.featurize_tokens(toks)
    assert abs(np.linalg.norm(val) - 1) < 1e-9
    idx2, _ = f.featurize_tokens(toks, keep=np.array([False, False, True]))
    assert set(idx2) <= set(idx)
    shares = f.token_feature_shares(toks)
    for lst in shares.values():
        assert abs(sum(w for _, w in lst) - 1) < 1e-9


def test_rationale_parsing():
    toks, mask = parse_rationale("[[muslims]] are [[a disease]]")
    assert toks == ["muslims", "are", "a", "disease"] and mask == [1, 0, 1, 1]


def test_counterfactual_keeps_template():
    cfg = GenConfig()
    spec = dict(lang="en", group="women", nn=0, na=0, x=0, tkey="H_DEROG", tidx=0, seed=3, label_true=1)
    a, _ = render_text(spec, cfg)
    b, _ = render_text(dict(spec, group="sikhs"), cfg)
    assert a[1:] == b[1:] and a[0] != b[0]


# ------------------------------------------------------------------ rewards
def test_reward_components():
    cfg = R.RewardConfig()
    r, c = R.total_reward(1, 1, np.array([1, 0, 1]), np.array([1, 0, 1]), True, 0.9, 0.9, 0.1, cfg)
    assert c["acc"] == 1 and c["plaus"] == 1 and c["fair"] == 0 and c["faith"] > 0.7
    r2, c2 = R.total_reward(1, 0, np.array([1, 0, 0]), None, True, 0.9, 0.9, 0.1, cfg)
    assert c2["fair"] == -1 and r2 < r
    _, c3 = R.total_reward(1, 1, np.array([0, 0, 0]), None, False, 0.9, 0.9, 0.9, cfg)
    assert c3["faith"] == -1  # hateful decision without evidence


def test_parse_llm_completion():
    p = R.parse_completion('sure: {"label": "hateful", "target": "religion", "rationale": ["vermin"], "explanation": "x"}')
    assert p["ok"] and p["label"] == 1 and p["rationale"] == ["vermin"]
    q = R.parse_completion("I think this is not hateful")
    assert not q["ok"] and q["label"] == 0


def test_llm_reward_funcs_conversational_format():
    from llm import reward_funcs as RF
    comps = [[{"role": "assistant", "content": '{"label":"hateful","rationale":["vermin"]}'}],
             [{"role": "assistant", "content": '{"label":"not_hateful","rationale":[]}'}]]
    kw = dict(label=[1, 1], tokens=[["they", "are", "vermin"]] * 2, rationale=[[0, 0, 1]] * 2,
              has_gold_rationale=[True, True], has_identity=[False, False])
    assert RF.correctness_reward(comps, **kw) == [1.0, -1.0]
    assert RF.plausibility_reward(comps, **kw)[0] > 0
    assert all(v == 0.5 for v in RF.format_reward(comps))


# ------------------------------------------------------------------ model
def _tiny_batch(n=5, dT=64, dV=7, dC=32, seed=0):
    rng = np.random.default_rng(seed)
    XT = sp.random(n, dT, density=0.2, random_state=seed, format="csr")
    XC = sp.random(n, dC, density=0.2, random_state=seed + 1, format="csr")
    return XT, rng.normal(size=(n, dV)), XC, rng.normal(size=(n, 6))


def test_backward_matches_finite_differences():
    XT, XV, XC, XM = _tiny_batch()
    m = FusionPolicy(64, 7, 32, 6, 16, h_text=8, h_img=4, h_ctx=4, h_fuse=6, seed=1)
    y = np.array([0, 1, 1, 0, 1])

    def loss():
        lg, _ = m.forward(XT, XV, XC, XM)
        P = softmax(lg)
        return -np.log(P[np.arange(5), y]).mean()

    lg, cache = m.forward(XT, XV, XC, XM)
    g = m.backward(cache, (softmax(lg) - np.eye(2)[y]) / 5)
    rng = np.random.default_rng(0)
    for k in ("WT", "WV", "WC", "WM", "WF", "WO", "bT", "bF"):
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in m.p[k].shape)
            if k == "WT" and XT[:, idx[0]].nnz == 0:
                continue
            old = m.p[k][idx]
            m.p[k][idx] = old + 1e-5
            lp = loss()
            m.p[k][idx] = old - 1e-5
            lm = loss()
            m.p[k][idx] = old
            num = (lp - lm) / 2e-5
            assert abs(num - g[k][idx]) < 1e-5 + 1e-3 * abs(num), (k, num, g[k][idx])


def test_modality_shapley_efficiency():
    XT, XV, XC, XM = _tiny_batch()
    m = FusionPolicy(64, 7, 32, 6, 16, h_text=8, h_img=4, h_ctx=4, h_fuse=6, seed=2)
    b = Batch(XT, XV, XC, XM, [["a"]] * 5, [sp.csr_matrix((1, 16))] * 5, np.zeros(5, int), [np.zeros(1)] * 5,
              np.zeros(5, bool), np.zeros(5, bool), [{}] * 5)
    phi, base = modality_shapley(m, b, np.arange(5))
    lg, _ = m.forward(XT, XV, XC, XM)
    assert np.allclose(phi.sum(1) + base, lg[:, 1] - lg[:, 0])


# ------------------------------------------------------------------ metrics & loaders
def test_token_f1_and_bias_metrics():
    assert token_f1([1, 0, 1], [1, 0, 0]) == 2 / 3
    recs = [{"group": g} for g in ["a", "a", "b", "b", None, None]]
    out = bias_metrics(recs, [0, 1, 0, 1, 0, 1], [0.2, 0.9, 0.6, 0.8, 0.1, 0.7])
    assert out["per_group"]["b"]["fpr"] == 1.0 and out["identity_fpr"] == 0.5


def test_hatexplain_loader(tmp_path):
    data = {
        "p1": {"post_tokens": ["those", "people", "are", "vermin"],
               "annotators": [{"label": "hatespeech", "target": ["Islam"]}] * 3,
               "rationales": [[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]},
        "p2": {"post_tokens": ["nice", "day"], "annotators": [{"label": "normal", "target": ["None"]}] * 3,
               "rationales": []},
    }
    (tmp_path / "d.json").write_text(json.dumps(data))
    (tmp_path / "s.json").write_text(json.dumps({"train": ["p1"], "val": [], "test": ["p2"]}))
    out = loaders.load_hatexplain(tmp_path / "d.json", tmp_path / "s.json")
    r = out["train"][0]
    assert r["label"] == 1 and r["rationale"] == [0, 0, 0, 1] and r["group"] == "muslims"
    assert out["test"][0]["label"] == 0
