"""Step 1: train the target model and save the baseline 'fingerprint'."""
import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_gen import make_data, FEATURES

N_BINS = 10


def entropy(proba):
    p = np.clip(proba, 1e-9, 1)
    return float(-(p * np.log2(p)).sum(axis=1).mean())  # max 1.0 for 2 classes


def main():
    X, y = make_data(20000, seed=1)
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.25, random_state=0)

    medians = X_tr.median().to_dict()
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=20,
                                   random_state=0, n_jobs=-1)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_ho)
    pred = proba.argmax(axis=1)
    ref_acc = float((pred == y_ho).mean())

    # ---- Baseline fingerprint: SUMMARIES ONLY, no raw rows ----
    feats = {}
    for f in FEATURES:
        col = X_tr[f].dropna().values
        edges = np.unique(np.quantile(col, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(col, bins=edges)
        feats[f] = {
            "edges": [float(e) if np.isfinite(e) else None for e in edges],
            "props": (counts / counts.sum()).tolist(),
            "mean": float(col.mean()), "std": float(col.std()),
            "lo": float(np.quantile(col, 0.005)), "hi": float(np.quantile(col, 0.995)),
            "missing_rate": float(X_tr[f].isna().mean()),
        }
    baseline = {
        "features": feats,
        "medians": medians,
        "confidence_mean": float(proba.max(axis=1).mean()),
        "entropy_mean": entropy(proba),
        "positive_rate": float(pred.mean()),
        "reference_accuracy": ref_acc,   # used ONLY by the evaluation/backtest, never by the monitor
    }
    os.makedirs("model", exist_ok=True)
    os.makedirs("baseline", exist_ok=True)
    joblib.dump(model, "model/model.joblib")
    with open("baseline/baseline.json", "w") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"Model trained. Holdout accuracy: {ref_acc:.3f}")
    print("Saved model/model.joblib and baseline/baseline.json")


if __name__ == "__main__":
    main()
