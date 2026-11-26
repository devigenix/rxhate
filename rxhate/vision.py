"""Lightweight visual encoder for meme images (V_i).

Offline, dependency-free descriptor: joint colour histogram + HOG-style gradient
orientation histograms + coarse spatial colour layout.  In the full system this is
swapped for a frozen CLIP/SigLIP image tower (see llm/ and README); the fusion model
only assumes a fixed-size dense vector per image.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def color_histogram(arr, bins=4):
    q = np.clip((arr / 256.0 * bins).astype(int), 0, bins - 1)
    code = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    h = np.bincount(code.ravel(), minlength=bins ** 3).astype(np.float64)
    return h / h.sum()


def hog(gray, cells=4, nbins=9):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)
    b = np.minimum((ang / (180.0 / nbins)).astype(int), nbins - 1)
    H, W = gray.shape
    ch, cw = H // cells, W // cells
    feats = []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            bb = b[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            h = np.bincount(bb, weights=m, minlength=nbins)
            feats.append(h / (np.linalg.norm(h) + 1e-6))
    return np.concatenate(feats)


def spatial_color(arr, grid=3):
    H, W, _ = arr.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            out.append(arr[i * H // grid:(i + 1) * H // grid, j * W // grid:(j + 1) * W // grid].reshape(-1, 3).mean(0) / 255.0)
    return np.concatenate(out)


def dark_red_mass(arr):
    """Fraction of dark / saturated-red pixels: a cheap colour-saliency cue."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    dark = (arr.max(-1) < 70).mean()
    red = ((r > 80) & (r < 170) & (g < 60) & (b < 70)).mean()
    return np.array([dark, red])


def image_features(path):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    arr = arr[:-18]  # drop caption banner (pseudo-text, carries no signal)
    gray = arr.mean(-1)
    return np.concatenate([color_histogram(arr), hog(gray), spatial_color(arr), dark_red_mass(arr)])


def batch_image_features(paths, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        return np.load(cache_file)
    F = np.stack([image_features(p) for p in paths])
    if cache_file:
        np.save(cache_file, F)
    return F
