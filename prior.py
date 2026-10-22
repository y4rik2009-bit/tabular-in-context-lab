"""On-the-fly mixture of synthetic binary classification mechanisms."""

from __future__ import annotations

import numpy as np
import torch


def sample_task(rng: np.random.Generator, n_context: int, n_query: int,
                max_features: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_features = int(rng.integers(3, max_features + 1))
    n = n_context + n_query
    latent = rng.normal(size=(n, n_features))
    mixing = rng.normal(size=(n_features, n_features))
    x = (latent @ mixing) / np.sqrt(n_features)
    x = (x - x[:n_context].mean(axis=0)) / (x[:n_context].std(axis=0) + 0.1)
    mechanism = int(rng.integers(4))
    w = rng.normal(size=n_features)
    w[rng.random(n_features) < 0.4] = 0
    if not np.any(w):
        w[0] = 1
    if mechanism == 0:  # sparse linear boundary
        score = x @ w
    elif mechanism == 1:  # nonlinear interaction
        score = (x @ w) * 0.4 + rng.choice([-1, 1]) * x[:, 0] * x[:, 1]
    elif mechanism == 2:  # radial region
        center = rng.normal(size=n_features)
        score = 1.5 - np.mean((x - center) ** 2, axis=1)
    else:  # periodic nonlinear boundary
        score = np.sin(x @ w) + 0.3 * x[:, 0]
    score = (score - np.median(score)) / (np.std(score) + 1e-4)
    temperature = rng.uniform(0.25, 1.1)
    probability = 1 / (1 + np.exp(-np.clip(score / temperature, -20, 20)))
    y = rng.binomial(1, probability).astype(np.int64)
    if rng.random() < 0.5:
        y = 1 - y
    permutation = rng.permutation(n_features)
    x = x[:, permutation]
    padded = np.zeros((n, max_features), dtype=np.float32)
    padded[:, :n_features] = np.clip(x, -5, 5).astype(np.float32)
    return padded, y, n_features


def sample_batch(rng: np.random.Generator, batch_size: int, n_context: int,
                 n_query: int, max_features: int):
    tasks = [sample_task(rng, n_context, n_query, max_features)
             for _ in range(batch_size)]
    x = torch.from_numpy(np.stack([t[0] for t in tasks]))
    y = torch.from_numpy(np.stack([t[1] for t in tasks]))
    features = torch.tensor([t[2] for t in tasks], dtype=torch.long)
    return x, y, features
