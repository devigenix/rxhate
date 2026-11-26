"""Multimodal featurisation x_i = {T_i, V_i, C_i} -> model inputs."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .data.synthetic import COMMUNITIES, GROUPS
from .text import HashedFeaturizer, RationaleTokenFeaturizer, normalize_token, tokenize
from .vision import batch_image_features

IMAGE_DIM = 64 + 144 + 27 + 2  # colour hist + HOG + spatial colour + saliency (see vision.py)
IDENTITY_TERMS = set()
for _, (_, en, hi) in GROUPS.items():
    IDENTITY_TERMS.update(en.split()[:1])
    IDENTITY_TERMS.update(hi.split()[:1])
IDENTITY_TERMS -= {"log", "people"}


def mentions_identity(tokens):
    return any(normalize_token(t) in IDENTITY_TERMS for t in tokens)


@dataclass
class Batch:
    XT: sp.csr_matrix      # text features (N x D_T)
    XV: np.ndarray         # image features (N x d_V), standardised
    XC: sp.csr_matrix      # parent-post features (N x D_C)
    XM: np.ndarray         # context metadata (N x 6)
    tokens: list           # list[list[str]]
    R: list                # per-post rationale token features (n_tokens x D_R csr)
    y: np.ndarray
    gold: list             # gold rationale masks
    has_gold: np.ndarray
    identity: np.ndarray   # post mentions an identity term
    records: list

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = np.asarray(idx)
        return Batch(self.XT[idx], self.XV[idx], self.XC[idx], self.XM[idx],
                     [self.tokens[i] for i in idx], [self.R[i] for i in idx], self.y[idx],
                     [self.gold[i] for i in idx], self.has_gold[idx], self.identity[idx],
                     [self.records[i] for i in idx])


class MultimodalFeaturizer:
    def __init__(self, d_text=2 ** 15, d_ctx=2 ** 13, d_rat=2 ** 15):
        self.text = HashedFeaturizer(dim=d_text, namespace="t")
        self.ctx = HashedFeaturizer(dim=d_ctx, namespace="c")
        self.rat = RationaleTokenFeaturizer(dim=d_rat)
        self.v_mean = None
        self.v_std = None

    def fit(self, XV_raw):
        self.v_mean = XV_raw.mean(0)
        self.v_std = XV_raw.std(0) + 1e-6
        return self

    def meta(self, records):
        M = np.zeros((len(records), len(COMMUNITIES) + 1))
        for i, r in enumerate(records):
            M[i, COMMUNITIES.index(r["community"])] = 1.0
            M[i, -1] = r["thread_toxicity"]
        return M

    def build(self, records, data_dir, cache_name=None, XV_raw=None, use_cf=False):
        tokens = [(r["cf_tokens"] if (use_cf and r.get("cf_tokens")) else r["tokens"]) for r in records]
        XT = self.text.transform(tokens)
        XC = self.ctx.transform([tokenize(r["parent_text"]) for r in records])
        if XV_raw is None:
            cache = os.path.join(data_dir, f"vfeat_{cache_name}.npy") if cache_name else None
            if all(r.get("image_path") for r in records):
                XV_raw = batch_image_features([os.path.join(data_dir, r["image_path"]) for r in records], cache)
            else:  # text-only corpora: the image modality is absent (zero input, as in modality dropout)
                XV_raw = np.zeros((len(records), IMAGE_DIM))
                has = [i for i, r in enumerate(records) if r.get("image_path")]
                if has:
                    XV_raw[has] = batch_image_features([os.path.join(data_dir, records[i]["image_path"]) for i in has])
        if self.v_mean is None:
            self.fit(XV_raw)
        XV = (XV_raw - self.v_mean) / self.v_std
        R = [self.rat.transform_post(t) for t in tokens]
        y = np.array([r["label"] for r in records])
        gold = [np.array(r["rationale"]) if not use_cf else np.zeros(len(t)) for r, t in zip(records, tokens)]
        has_gold = np.array([bool(r.get("has_gold_rationale", True)) and not use_cf for r in records])
        ident = np.array([mentions_identity(t) for t in tokens])
        return Batch(XT, XV, XC, self.meta(records), tokens, R, y, gold, has_gold, ident, records), XV_raw

    def text_matrix(self, token_lists, keeps=None):
        return self.text.transform(token_lists, keeps)
