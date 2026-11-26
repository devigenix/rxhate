"""Text normalisation and hashed n-gram featurisation for multilingual / code-mixed posts.

Low-resource, code-mixed text breaks word-level vocabularies (spelling variants,
transliteration, leetspeak obfuscation).  We therefore
  1. normalise (lower-case, de-leet, squeeze elongations, map romanised-Hindi variants),
  2. featurise with *signed feature hashing* (Weinberger et al., 2009) over word uni/bi-grams
     and within-word character 2-4-grams, which needs no vocabulary and degrades gracefully.
Every feature remembers which token(s) produced it, so attributions computed on the
feature vector can be mapped back to tokens (needed by the XAI module and by the
rationale-faithfulness rewards).
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

LEET_MAP = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "0": "o", "$": "s"})
VARIANT_MAP = {
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi",
    "hn": "hain",
    "baahar": "bahar", "bhar": "bahar",
    "yr": "yaar", "yarr": "yaar",
    "bohot": "bahut", "bhot": "bahut", "bahot": "bahut",
    "kia": "kya", "logg": "log", "mai": "mein",
    "ache": "acche", "achhe": "acche",
}
_ELONG = re.compile(r"(.)\1{2,}")


def normalize_token(tok: str) -> str:
    t = tok.lower()
    if any(c.isalpha() for c in t):
        t = t.translate(LEET_MAP)
    t = _ELONG.sub(r"\1\1", t)
    return VARIANT_MAP.get(t, t)


def tokenize(text: str):
    return re.findall(r"[#@$\w']+|[^\w\s]", text.lower())


def _h(s: str, dim: int):
    v = zlib.crc32(s.encode("utf-8"))
    return v % dim, (1.0 if (v >> 31) & 1 else -1.0)


@dataclass
class HashedFeaturizer:
    dim: int = 2 ** 15
    char_ngrams: tuple = (2, 3, 4)
    word_bigrams: bool = True
    namespace: str = "t"
    _cache: dict = field(default_factory=dict, repr=False)

    def token_features(self, tok: str):
        """Features that belong to a single token (word unigram + char n-grams)."""
        key = tok
        if key in self._cache:
            return self._cache[key]
        nt = normalize_token(tok)
        feats = {}
        i, s = _h(f"{self.namespace}w:{nt}", self.dim)
        feats[i] = feats.get(i, 0.0) + s
        padded = f"<{nt}>"
        for n in self.char_ngrams:
            for k in range(len(padded) - n + 1):
                i, s = _h(f"{self.namespace}c{n}:{padded[k:k + n]}", self.dim)
                feats[i] = feats.get(i, 0.0) + s * 0.5
        out = (np.fromiter(feats.keys(), dtype=np.int64), np.fromiter(feats.values(), dtype=np.float64))
        self._cache[key] = out
        return out

    def bigram_feature(self, a: str, b: str):
        return _h(f"{self.namespace}b:{normalize_token(a)}_{normalize_token(b)}", self.dim)

    def featurize_tokens(self, tokens, keep=None):
        """Return (indices, values) of the L2-normalised hashed vector for `tokens`.
        `keep` (bool array) restricts to a subset of tokens: bigrams survive only if both
        tokens are kept. This is what the sufficiency/comprehensiveness rewards use."""
        n = len(tokens)
        if keep is None:
            keep = np.ones(n, dtype=bool)
        idx_list, val_list = [], []
        for t in range(n):
            if keep[t]:
                i, v = self.token_features(tokens[t])
                idx_list.append(i)
                val_list.append(v)
        if self.word_bigrams:
            for t in range(n - 1):
                if keep[t] and keep[t + 1]:
                    i, s = self.bigram_feature(tokens[t], tokens[t + 1])
                    idx_list.append(np.array([i]))
                    val_list.append(np.array([s]))
        if not idx_list:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.concatenate(idx_list)
        val = np.concatenate(val_list)
        uidx, inv = np.unique(idx, return_inverse=True)
        uval = np.zeros(len(uidx))
        np.add.at(uval, inv, val)
        nrm = np.linalg.norm(uval)
        if nrm > 0:
            uval /= nrm
        return uidx, uval

    def token_feature_shares(self, tokens):
        """For mapping feature attributions back to tokens.
        Returns dict feature_index -> list[(token_position, share)] with shares summing to 1."""
        contrib = {}
        for t, tok in enumerate(tokens):
            idx, val = self.token_features(tok)
            for i, v in zip(idx, val):
                contrib.setdefault(int(i), []).append((t, abs(v)))
        if self.word_bigrams:
            for t in range(len(tokens) - 1):
                i, _ = self.bigram_feature(tokens[t], tokens[t + 1])
                contrib.setdefault(int(i), []).extend([(t, 0.5), (t + 1, 0.5)])
        shares = {}
        for i, lst in contrib.items():
            tot = sum(w for _, w in lst) or 1.0
            agg = {}
            for t, w in lst:
                agg[t] = agg.get(t, 0.0) + w / tot
            shares[i] = list(agg.items())
        return shares

    def transform(self, token_lists, keeps=None):
        rows, cols, vals = [], [], []
        for r, toks in enumerate(token_lists):
            idx, val = self.featurize_tokens(toks, None if keeps is None else keeps[r])
            rows.append(np.full(len(idx), r))
            cols.append(idx)
            vals.append(val)
        if rows:
            rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(token_lists), self.dim))


@dataclass
class RationaleTokenFeaturizer:
    """Per-token features for the rationale head: the token itself plus its neighbours
    (so 'calling X parasites' can be treated differently from 'X are parasites')."""
    dim: int = 2 ** 15
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.base = HashedFeaturizer(dim=self.dim, namespace="r", word_bigrams=False)

    def transform_post(self, tokens):
        n = len(tokens)
        rows, cols, vals = [], [], []
        norm = [normalize_token(t) for t in tokens]
        for t in range(n):
            idx, val = self.base.token_features(tokens[t])
            feats = dict(zip(idx.tolist(), val.tolist()))
            ctx = [("L1", norm[t - 1] if t > 0 else "<s>"), ("R1", norm[t + 1] if t < n - 1 else "</s>"),
                   ("L2", norm[t - 2] if t > 1 else "<s>"), ("R2", norm[t + 2] if t < n - 2 else "</s>"),
                   ("POS", "first" if t == 0 else ("last" if t == n - 1 else "mid"))]
            for k, w in ctx:
                i, s = _h(f"r{k}:{w}", self.dim)
                feats[i] = feats.get(i, 0.0) + 0.7 * s
            v = np.array(list(feats.values()))
            v = v / (np.linalg.norm(v) + 1e-9)
            rows.extend([t] * len(v))
            cols.extend(feats.keys())
            vals.extend(v.tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, self.dim))
