"""Step 2: simulate time passing and run the backtest.

The monitor never sees labels. Labels are used HERE ONLY, to measure the real accuracy
so we can check whether the warning really came before the accuracy dropped.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd

from data_gen import make_data
from monitor import check_batch, RiskTracker, load_baseline, WARN, level, diagnose

N_BATCHES, BATCH_SIZE = 40, 500
DRIFT_START, DRIFT_END = 10, 34        # drift ramps up between these batches
ACC_DROP = 0.05                        # "failure" = rolling accuracy falls 5 pts below reference
SCENARIOS = ["control", "gradual_drift", "quality_break"]


def schedule(scenario, i):
    """Returns (drift, missing_rate, unit_error_rate) for batch i."""
    ramp = float(np.clip((i - DRIFT_START) / (DRIFT_END - DRIFT_START), 0, 1))
    if scenario == "gradual_drift":
        return ramp, 0.0, 0.0
    if scenario == "quality_break":
        return 0.0, 0.1 * ramp, 0.6 * ramp
    return 0.0, 0.0, 0.0


def run(scenario, seed, model, baseline):
    tracker, rows = RiskTracker(), []
    for i in range(N_BATCHES):
        drift, miss, unit = schedule(scenario, i)
        X, y = make_data(BATCH_SIZE, seed=seed * 1000 + i + 7, drift=drift, missing_rate=miss, unit_error_rate=unit)
        X_model = X.fillna(pd.Series(baseline["medians"]))      # normal pipeline imputation
        proba = model.predict_proba(X_model)

        res = check_batch(X, proba, baseline)                    # <- NO labels passed in
        risk, eta = tracker.update(res["risk_raw"])

        cause_text, action, main_cause = diagnose(res)
        acc = float((proba.argmax(axis=1) == y).mean())          # evaluation only
        rows.append({"batch": i, "scenario": scenario, "seed": seed,
                     "risk": risk, "level": level(risk), "eta_batches": eta,
                     "main_cause": main_cause, "diagnosis": cause_text, "action": action,
                     "drift_score": res["drift_score"], "quality_score": res["quality_score"],
                     "behavior_score": res["behavior_score"], "confidence": res["confidence"],
                     "entropy": res["entropy"], "positive_rate": res["positive_rate"],
                     "missing_increase": res["missing_increase"],
                     "accuracy": acc, "true_drift": drift, "true_missing": miss, "true_unit_error": unit,
                     **{f"psi_{k}": v for k, v in res["psi"].items()}})
    df = pd.DataFrame(rows)
    df["acc_roll"] = df["accuracy"].rolling(3, min_periods=3).mean()
    return df


def evaluate(df, ref_acc):
    """Find first failure batch and first alert batch -> lead time."""
    fail = df.index[(df["acc_roll"] < ref_acc - ACC_DROP)]
    fail_batch = int(df.loc[fail[0], "batch"]) if len(fail) else None
    hot = (df["risk"] >= WARN).astype(int)
    alert = df.index[(hot.rolling(2).sum() == 2)]
    alert_batch = int(df.loc[alert[0], "batch"]) if len(alert) else None
    lead = (fail_batch - alert_batch) if (fail_batch is not None and alert_batch is not None) else None
    return {"failure_batch": fail_batch, "alert_batch": alert_batch, "lead_time": lead}


def main(n_seeds=10):
    model = joblib.load("model/model.joblib")
    baseline = load_baseline()
    ref = baseline["reference_accuracy"]
    os.makedirs("results", exist_ok=True)

    summary, detail = [], []
    for sc in SCENARIOS:
        for seed in range(n_seeds):
            df = run(sc, seed, model, baseline)
            ev = evaluate(df, ref)
            summary.append({"scenario": sc, "seed": seed, **ev})
            if seed == 0:
                detail.append(df)
    pd.concat(detail).to_csv("results/results.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv("results/backtest_runs.csv", index=False)

    out = {"reference_accuracy": ref, "n_seeds": n_seeds, "scenarios": {}}
    for sc, g in s.groupby("scenario"):
        if sc == "control":
            out["scenarios"][sc] = {"runs": len(g), "false_alarm_rate": float(g["alert_batch"].notna().mean())}
        elif sc == "gradual_drift":
            ok = g["failure_batch"].notna()
            caught = g[ok & g["alert_batch"].notna() & (g["lead_time"] > 0)]
            out["scenarios"][sc] = {
                "runs": len(g), "runs_with_real_failure": int(ok.sum()),
                "warned_before_failure": int(len(caught)),
                "median_lead_time_batches": float(caught["lead_time"].median()) if len(caught) else None,
                "min_lead_time": float(caught["lead_time"].min()) if len(caught) else None}
        else:  # quality_break: diagnosis demo, accuracy barely moves in this setup
            out["scenarios"][sc] = {
                "runs": len(g), "alerts_raised": int(g["alert_batch"].notna().sum()),
                "runs_with_accuracy_failure": int(g["failure_batch"].notna().sum()),
                "note": "Broken feed is caught and diagnosed as a data-quality issue; "
                        "accuracy impact is small in this synthetic setup, so no lead-time claim is made."}
    with open("results/summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


def simulate_text_demo():
    """Demonstrate model-agnostic drift monitoring on text data using TextExtractor.

    1. Generates 500 normal short product reviews (reference set).
    2. Trains a text classifier (TfidfVectorizer + LogisticRegression).
    3. Builds reference baseline fingerprint using TextExtractor.
    4. Evaluates reference batch (control test) -> low risk/drift.
    5. Generates 500 drifted texts (longer, uppercase, numbers, gibberish).
    6. Evaluates drifted batch -> high drift score and flags shifted features.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from data_gen import make_text_data
    from extractors import get_extractor
    from monitor import build_baseline, check_batch, level, diagnose

    print("=" * 60)
    print("AERO Text Monitoring Demo (Model-Agnostic Engine)")
    print("=" * 60)

    # 1. Reference Data & Text Model
    print("Generating reference text dataset (~500 reviews)...")
    texts_ref, y_ref = make_text_data(n=500, seed=42, drifted=False)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("lr", LogisticRegression(random_state=42))
    ])
    clf.fit(texts_ref, y_ref)
    proba_ref = clf.predict_proba(texts_ref)

    # 2. Build Baseline using TextExtractor
    extractor = get_extractor("text")
    baseline = build_baseline(texts_ref, proba_ref, y=y_ref, extractor=extractor)

    # 3. Check Reference Batch
    res_clean = check_batch(texts_ref, proba_ref, baseline, extractor=extractor)
    print("\n--- 1. Reference Batch (Normal Reviews) ---")
    print(f"Drift Score: {res_clean['drift_score']:.1f} / 100")
    print(f"Risk Score : {res_clean['risk_raw']:.1f} / 100 ({level(res_clean['risk_raw'])})")

    # 4. Check Drifted Batch
    print("\nGenerating drifted text dataset (~500 drifted texts)...")
    texts_drift, y_drift = make_text_data(n=500, seed=99, drifted=True)
    proba_drift = clf.predict_proba(texts_drift)
    res_drift = check_batch(texts_drift, proba_drift, baseline, extractor=extractor)
    diag_text, action, cause = diagnose(res_drift)

    print("\n--- 2. Drifted Batch (Long, Numeric, Uppercase, Gibberish) ---")
    print(f"Drift Score: {res_drift['drift_score']:.1f} / 100")
    print(f"Risk Score : {res_drift['risk_raw']:.1f} / 100 ({level(res_drift['risk_raw'])})")
    print(f"Diagnosis  : {diag_text}")
    print(f"Action     : {action}")
    print("\nFeature Drift Breakdown (PSI):")
    for feat, psi_val in sorted(res_drift["psi"].items(), key=lambda kv: -kv[1]):
        status = "LARGE" if psi_val > 0.25 else ("MODERATE" if psi_val > 0.1 else "STABLE")
        print(f"  - {feat:<20}: PSI = {psi_val:.4f} [{status}]")
    print("=" * 60)
    return res_clean, res_drift


def simulate_image_demo():
    """Demonstrate model-agnostic drift monitoring on image data using ImageExtractor.

    1. Loads reference digit images and splits into training (1,000 samples)
       and a FRESH unseen control batch (different split, not training data).
    2. Trains a LogisticRegression classifier on flattened reference images.
    3. Builds reference baseline using ImageExtractor.
    4. Evaluates the FRESH normal batch (control) -> low drift / no false alarms.
    5. Generates the drifted batch (mixed corruption: darker, noise, blur).
    6. Evaluates the drifted batch -> flags high drift, identifies shifted features,
       and displays the drop in model accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from data_gen import make_image_data
    from extractors import get_extractor
    from monitor import build_baseline, check_batch, level, diagnose

    print("=" * 60)
    print("AERO Image Monitoring Demo (Model-Agnostic Engine)")
    print("=" * 60)

    # 1. Reference Data: train split + fresh control split
    imgs_all, y_all = make_image_data(drifted=False, seed=42)
    X_train, y_train = imgs_all[:1000], y_all[:1000]
    X_control, y_control = imgs_all[1000:], y_all[1000:]

    # 2. Train LogisticRegression on flattened images
    print("Training LogisticRegression on flattened reference digits (1,000 samples)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train.reshape(len(X_train), -1), y_train)

    proba_train = clf.predict_proba(X_train.reshape(len(X_train), -1))
    acc_train = float((clf.predict(X_train.reshape(len(X_train), -1)) == y_train).mean())
    print(f"  - Classifier training accuracy: {acc_train * 100:.1f}%")

    # 3. Build Baseline using ImageExtractor
    extractor = get_extractor("image")
    baseline = build_baseline(X_train, proba_train, y=y_train, extractor=extractor)

    # 4. Check Fresh Normal Control Batch (Unseen Test Split)
    print(f"\n--- 1. Fresh Normal Batch ({len(X_control)} unseen digits - Control) ---")
    proba_control = clf.predict_proba(X_control.reshape(len(X_control), -1))
    acc_control = float((clf.predict(X_control.reshape(len(X_control), -1)) == y_control).mean())
    res_control = check_batch(X_control, proba_control, baseline, extractor=extractor)
    print(f"Accuracy   : {acc_control * 100:.1f}%")
    print(f"Drift Score: {res_control['drift_score']:.1f} / 100")
    print(f"Risk Score : {res_control['risk_raw']:.1f} / 100 ({level(res_control['risk_raw'])})")

    # 5. Check Drifted Batch (Mixed Corruption: Darker + Blur + Noise)
    print(f"\n--- 2. Drifted Batch (Corrupted: Darker, Noise, Blur) ---")
    X_drift, y_drift = make_image_data(drifted=True, seed=99, corruption="mixed")
    proba_drift = clf.predict_proba(X_drift.reshape(len(X_drift), -1))
    acc_drift = float((clf.predict(X_drift.reshape(len(X_drift), -1)) == y_drift).mean())
    res_drift = check_batch(X_drift, proba_drift, baseline, extractor=extractor)
    diag_text, action, cause = diagnose(res_drift)

    print(f"Accuracy   : {acc_drift * 100:.1f}% (dropped from {acc_control * 100:.1f}%)")
    print(f"Drift Score: {res_drift['drift_score']:.1f} / 100")
    print(f"Risk Score : {res_drift['risk_raw']:.1f} / 100 ({level(res_drift['risk_raw'])})")
    print(f"Diagnosis  : {diag_text}")
    print(f"Action     : {action}")
    print("\nFeature Drift Breakdown (PSI):")
    for feat, psi_val in sorted(res_drift["psi"].items(), key=lambda kv: -kv[1]):
        status = "LARGE" if psi_val > 0.25 else ("MODERATE" if psi_val > 0.1 else "STABLE")
        print(f"  - {feat:<20}: PSI = {psi_val:.4f} [{status}]")
    print("=" * 60)
    return res_control, res_drift


if __name__ == "__main__":
    import sys
    if "--text" in sys.argv:
        simulate_text_demo()
    elif "--image" in sys.argv:
        simulate_image_demo()
    else:
        main()
