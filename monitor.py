"""The monitoring engine (the heart of the project).

IMPORTANT: nothing in this file ever receives true labels.
It only sees (1) the incoming feature batch and (2) the model's own outputs.
"""
import json
import numpy as np
import pandas as pd

WEIGHTS = {"drift": 0.45, "quality": 0.20, "behavior": 0.35}
WARN, HIGH, DANGER = 35, 55, 75       # risk thresholds (0-100)
EMA_ALPHA = 0.5                        # smoothing so one noisy batch doesn't trigger alerts


def load_baseline(path="baseline/baseline.json"):
    with open(path) as fh:
        return json.load(fh)


def _psi(expected, actual, eps=1e-4):
    e = np.clip(np.asarray(expected), eps, None)
    a = np.clip(np.asarray(actual), eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def _entropy(proba):
    p = np.clip(proba, 1e-9, 1)
    return float(-(p * np.log2(p)).sum(axis=1).mean())


def check_batch(X, proba, baseline, extractor=None):
    """Score one batch.

    Parameters
    ----------
    X : Any
        Feature DataFrame or raw inputs (e.g. list of text strings).
        If an extractor is passed, X is transformed into a numeric DataFrame.
    proba : np.ndarray
        Model's predicted probabilities.
    baseline : dict
        Baseline fingerprint dictionary.
    extractor : FeatureExtractor or str, optional
        Extractor instance or data type ('tabular', 'text').
    """
    if extractor is not None:
        if isinstance(extractor, str):
            from extractors import get_extractor
            extractor = get_extractor(extractor)
        if hasattr(extractor, "max_value") and extractor.max_value is None and "max_value" in baseline:
            extractor.max_value = baseline["max_value"]
        X = extractor.extract(X)
    elif "extractor" in baseline and baseline["extractor"] is not None:
        extractor = baseline["extractor"]
        X = extractor.extract(X)
    elif not isinstance(X, pd.DataFrame):
        from extractors import TabularExtractor
        X = TabularExtractor().extract(X)

    feats = baseline["features"]

    # ---- 1. Distribution drift (PSI per feature) ----
    psi = {}
    for f, b in feats.items():
        if f not in X.columns:
            continue
        col = X[f].dropna().values
        edges = np.array([-np.inf if e is None and i == 0 else (np.inf if e is None else e)
                          for i, e in enumerate(b["edges"])])
        counts, _ = np.histogram(col, bins=edges)
        psi[f] = _psi(b["props"], counts / max(counts.sum(), 1))
    top3 = sorted(psi.values(), reverse=True)[:3]
    drift_score = 100 * min(1.0, np.mean(top3) / 0.35) if top3 else 0.0

    # ---- 2. Data quality ----
    miss_list = [X[f].isna().mean() - b["missing_rate"] for f, b in feats.items() if f in X.columns]
    miss_inc = max(miss_list) if miss_list else 0.0
    out_inc = []
    for f, b in feats.items():
        if f not in X.columns:
            continue
        col = X[f].dropna()
        out_inc.append(((col < b["lo"]) | (col > b["hi"])).mean() - 0.01)
    quality_score = 100 * min(1.0, max(miss_inc / 0.15, max(out_inc or [0.0]) / 0.20, 0))

    # ---- 3. Model behaviour (no labels needed) ----
    proba = np.asarray(proba)
    conf = float(proba.max(axis=1).mean())
    ent = _entropy(proba)
    pred = proba.argmax(axis=1)
    if proba.ndim == 2 and proba.shape[1] > 2:
        n_classes = proba.shape[1]
        norm_ent = np.log2(n_classes)
        conf_drop = max(0.0, baseline.get("confidence_mean", conf) - conf)
        ent_rise = max(0.0, (ent - baseline.get("entropy_mean", ent)) / norm_ent)
        cur_dist = np.bincount(pred, minlength=n_classes) / max(len(pred), 1)
        base_dist = np.array(baseline.get("class_distribution", cur_dist))
        pos_shift = float(0.5 * np.sum(np.abs(cur_dist - base_dist)))
        pos_shift_threshold = 0.12 * np.sqrt(n_classes / 2.0)
        pos = float(pred.mean())
    else:
        pos = float(pred.mean())
        conf_drop = max(0.0, baseline.get("confidence_mean", conf) - conf)
        ent_rise = max(0.0, ent - baseline.get("entropy_mean", ent))
        pos_shift = abs(pos - baseline.get("positive_rate", pos))
        pos_shift_threshold = 0.12
    behavior_score = 100 * min(1.0, (min(1, conf_drop / 0.06) + min(1, ent_rise / 0.08)
                                     + min(1, pos_shift / pos_shift_threshold)) / 2.0)

    weighted = (WEIGHTS["drift"] * drift_score + WEIGHTS["quality"] * quality_score
                + WEIGHTS["behavior"] * behavior_score)
    # A single very strong signal (e.g. broken data feed) must not be diluted by calm ones.
    raw = max(weighted, 0.7 * max(drift_score, quality_score, behavior_score))
    return {"drift_score": drift_score, "quality_score": quality_score,
            "behavior_score": behavior_score, "risk_raw": raw,
            "confidence": conf, "entropy": ent, "positive_rate": pos,
            "psi": psi, "missing_increase": float(miss_inc)}


def build_baseline(X, proba, y=None, n_bins=10, extractor=None):
    """Build a baseline fingerprint dictionary from reference data.

    Works with tabular DataFrames or raw inputs processed via an extractor.
    """
    if extractor is not None:
        if isinstance(extractor, str):
            from extractors import get_extractor
            extractor = get_extractor(extractor)
        X = extractor.extract(X)
    elif not isinstance(X, pd.DataFrame):
        from extractors import TabularExtractor
        X = TabularExtractor().extract(X)

    medians = {c: float(v) for c, v in X.median().to_dict().items()}
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    ref_acc = float((pred == y).mean()) if y is not None else None
    class_dist = None
    if proba.ndim == 2 and proba.shape[1] > 2:
        class_dist = (np.bincount(pred, minlength=proba.shape[1]) / max(len(pred), 1)).tolist()

    feats = {}
    for f in X.columns:
        col = X[f].dropna().values
        if len(col) == 0:
            continue
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            val = float(col[0]) if len(col) > 0 else 0.0
            eps = 1e-4
            edges = np.array([-np.inf, val - eps, val + eps, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(col, bins=edges)
        feats[f] = {
            "edges": [float(e) if np.isfinite(e) else None for e in edges],
            "props": (counts / max(counts.sum(), 1)).tolist(),
            "mean": float(col.mean()),
            "std": float(col.std()) if len(col) > 1 else 0.0,
            "lo": float(np.quantile(col, 0.005)),
            "hi": float(np.quantile(col, 0.995)),
            "missing_rate": float(X[f].isna().mean()),
        }

    max_val = getattr(extractor, "max_value", None)
    return {
        "features": feats,
        "medians": medians,
        "confidence_mean": float(proba.max(axis=1).mean()),
        "entropy_mean": _entropy(proba),
        "positive_rate": float(pred.mean()),
        "class_distribution": class_dist,
        "reference_accuracy": ref_acc,
        "extractor": extractor,
        "max_value": max_val,
    }


class RiskTracker:
    """Smooths risk over time and forecasts when it will reach the DANGER level."""

    def __init__(self):
        self.history = []

    def update(self, raw):
        prev = self.history[-1] if self.history else raw
        smooth = EMA_ALPHA * raw + (1 - EMA_ALPHA) * prev
        self.history.append(smooth)
        return smooth, self.forecast()

    def forecast(self, window=6):
        """Linear trend over the last few batches -> batches until DANGER."""
        h = self.history[-window:]
        if len(h) < 4:
            return None
        slope = np.polyfit(np.arange(len(h)), h, 1)[0]
        if h[-1] >= DANGER:
            return 0.0
        if slope < 0.5:          # not rising meaningfully
            return None
        return float((DANGER - h[-1]) / slope)


def level(risk):
    if risk >= DANGER: return "DANGER"
    if risk >= HIGH: return "HIGH"
    if risk >= WARN: return "WATCH"
    return "HEALTHY"


def diagnose(result):
    """Turn the individual scores into a plain-English likely cause + action."""
    if level(result["risk_raw"]) == "HEALTHY":
        return ("No significant drift detected. The model inputs and outputs look normal.",
                "No action needed. Continue monitoring.", "NONE")

    parts = {"drift": result["drift_score"], "quality": result["quality_score"],
             "behavior": result["behavior_score"]}
    top_feats = [f for f, _ in sorted(result["psi"].items(), key=lambda kv: -kv[1])[:3]]
    main = max(parts, key=parts.get)
    if max(parts.values()) < WARN:
        return ("No significant drift detected. The model inputs and outputs look normal.",
                "No action needed. Continue monitoring.", "NONE")
    if main == "quality":
        return ("Likely a DATA-QUALITY problem (missing or out-of-range values).",
                "Check the data pipeline first; retraining is probably not needed.", "quality")
    if main == "drift":
        return (f"Likely real-world DRIFT. Most-shifted features: {', '.join(top_feats)}.",
                "Start collecting fresh labelled data and plan a retrain.", "drift")
    return ("The model is behaving differently (confidence/outputs changing).",
            "Route borderline predictions to human review and investigate.", "behavior")
