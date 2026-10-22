"""Compare a frozen in-context network with classical models on held-out UCI data."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import catboost
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from miniicl import MiniTabICL

DATASETS = {
    "banknote": {"id": 267, "positive": "1"},
    "breast_original": {"id": 15, "positive": "4"},
    "ionosphere": {"id": 52, "positive": "g"},
}


def load_dataset(name: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        from ucimlrepo import fetch_ucirepo

        raw = fetch_ucirepo(id=DATASETS[name]["id"])
        frame = raw.data.features.copy()
        target = raw.data.targets
        if target.shape[1] != 1:
            raise ValueError(f"{name} must have one target")
        frame["__target__"] = target.iloc[:, 0].to_numpy()
        frame.to_csv(path, index=False)
    if "__target__" not in frame:
        raise ValueError(f"Missing target in {path}")
    target = frame.pop("__target__").astype(str)
    y = (target == DATASETS[name]["positive"]).to_numpy(dtype=np.int64)
    if y.min() == y.max():
        raise ValueError(f"Target mapping produced one class for {name}: {target.unique()[:5]}")
    x = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(x).any():
        raise ValueError(f"No numeric features in {name}")
    return x, y


def make_split(x: np.ndarray, y: np.ndarray, seed: int, n_context: int,
               n_test: int, max_features: int):
    pool, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                  stratify=y, random_state=seed)
    if n_context > len(pool):
        raise ValueError("Context exceeds training pool")
    context, _ = train_test_split(pool, train_size=n_context, stratify=y[pool],
                                  random_state=seed + 100)
    rng = np.random.default_rng(seed + 200)
    test = np.sort(rng.choice(test, min(n_test, len(test)), replace=False))
    # All fitted preprocessing uses context only. Baselines receive the same columns.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xc = imputer.fit_transform(x[context])
    xe = imputer.transform(x[test])
    if xc.shape[1] > max_features:
        selector = SelectKBest(f_classif, k=max_features)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = selector.fit_transform(xc, y[context])
            xe = selector.transform(xe)
    mean, sd = xc.mean(0), xc.std(0)
    xc = np.clip((xc - mean) / (sd + 0.1), -5, 5).astype(np.float32)
    xe = np.clip((xe - mean) / (sd + 0.1), -5, 5).astype(np.float32)
    return xc, y[context], xe, y[test], context, test


@torch.inference_mode()
def predict_icl(model: MiniTabICL, xc: np.ndarray, yc: np.ndarray,
                xe: np.ndarray, max_features: int, batch_size: int = 16) -> np.ndarray:
    model.eval()
    n_context, n_features = xc.shape
    if n_features > max_features:
        raise ValueError("Too many features for pretrained checkpoint")
    output = []
    for start in range(0, len(xe), batch_size):
        queries = xe[start:start + batch_size]
        # Each task in the batch has exactly one query. No test row sees another.
        episode = np.zeros((len(queries), n_context + 1, max_features), dtype=np.float32)
        episode[:, :n_context, :n_features] = xc[None]
        episode[:, -1, :n_features] = queries
        labels = np.broadcast_to(yc, (len(queries), n_context)).copy()
        nf = torch.full((len(queries),), n_features, dtype=torch.long)
        logits = model(torch.from_numpy(episode), torch.from_numpy(labels), nf)
        output.extend(torch.sigmoid(logits[:, 0]).cpu().numpy())
    return np.asarray(output)


def metric_row(dataset: str, seed: int, method: str, y: np.ndarray,
               probabilities: np.ndarray, seconds: float,
               n_context: int, n_test: int) -> dict:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {"dataset": dataset, "seed": seed, "method": method,
            "n_context": n_context, "n_test": n_test,
            "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "seconds": float(seconds)}


def run_one(name: str, seed: int, args, model: MiniTabICL,
            random_model: MiniTabICL) -> list[dict]:
    x, y = load_dataset(name, Path(".cache/data"))
    xc, yc, xe, ye, _, _ = make_split(
        x, y, seed, args.context, args.max_test, args.features)
    rows = []
    start = time.perf_counter()
    pred = predict_icl(model, xc, yc, xe, args.features)
    rows.append(metric_row(name, seed, "MiniTabICL", ye, pred,
                           time.perf_counter() - start, len(yc), len(ye)))

    start = time.perf_counter()
    random_pred = predict_icl(random_model, xc, yc, xe, args.features)
    rows.append(metric_row(name, seed, "MiniTabICL, random weights", ye,
                           random_pred, time.perf_counter() - start,
                           len(yc), len(ye)))

    # A deliberately broken context: tests whether the network uses paired labels.
    rng = np.random.default_rng(seed + 999)
    shuffled = rng.permutation(yc)
    start = time.perf_counter()
    pred_shuffled = predict_icl(model, xc, shuffled, xe, args.features)
    rows.append(metric_row(name, seed, "MiniTabICL, shuffled context labels",
                           ye, pred_shuffled, time.perf_counter() - start,
                           len(yc), len(ye)))

    start = time.perf_counter()
    logreg = LogisticRegression(max_iter=1000, random_state=seed)
    logreg.fit(xc, yc)
    rows.append(metric_row(name, seed, "Logistic regression", ye,
                           logreg.predict_proba(xe)[:, 1],
                           time.perf_counter() - start, len(yc), len(ye)))

    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(xc, yc)
    rows.append(metric_row(name, seed, "5-NN", ye,
                           knn.predict_proba(xe)[:, 1],
                           time.perf_counter() - start, len(yc), len(ye)))

    start = time.perf_counter()
    cat = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                             loss_function="Logloss", verbose=False,
                             random_seed=seed, thread_count=4,
                             allow_writing_files=False)
    cat.fit(xc, yc)
    rows.append(metric_row(name, seed, "CatBoost", ye,
                           cat.predict_proba(xe)[:, 1],
                           time.perf_counter() - start, len(yc), len(ye)))
    print(f"{name} seed={seed}: " + ", ".join(
        f"{r['method']} AUC={r['auc']:.3f}" for r in rows), flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/model.pt")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--max-test", type=int, default=100)
    p.add_argument("--features", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MiniTabICL(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.manual_seed(999)
    random_model = MiniTabICL(**checkpoint["model_config"]).eval()
    start = time.perf_counter()
    rows = [row for name in args.datasets for seed in args.seeds
            for row in run_one(name, seed, args, model, random_model)]
    out = Path("results/raw")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics.csv", index=False)
    summary = frame.groupby(["dataset", "method"], sort=False)[
        ["auc", "log_loss", "brier", "accuracy", "seconds"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)
    manifest = {"args": vars(args), "wall_seconds": round(time.perf_counter() - start, 2),
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "catboost": catboost.__version__,
                "note": "Training uses only synthetic tasks. Each UCI split supplies context labels; no test label fits preprocessing or models."}
    (out / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(frame)} result rows", flush=True)


if __name__ == "__main__":
    main()
