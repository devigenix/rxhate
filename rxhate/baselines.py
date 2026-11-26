"""Classical baselines (scikit-learn)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text import normalize_token


def _norm_text(r):
    return " ".join(normalize_token(t) for t in r["tokens"])


def tfidf_lr(train, test, analyzer="word", ngram=(1, 2), C=4.0):
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform([_norm_text(r) for r in train])
    Xte = vec.transform([_norm_text(r) for r in test])
    clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, [r["label"] for r in train])
    return clf.predict_proba(Xte)[:, 1], (vec, clf)


def multimodal_lr(Btr, Bte, C=4.0):
    """Early-fusion linear model over [text | image | context | meta] features."""
    def stack(b):
        return sp.hstack([b.XT, sp.csr_matrix(b.XV * 0.1), b.XC, sp.csr_matrix(b.XM)]).tocsr()
    clf = LogisticRegression(C=C, max_iter=3000).fit(stack(Btr), Btr.y)
    return clf.predict_proba(stack(Bte))[:, 1], clf
