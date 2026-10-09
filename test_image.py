"""Test script for Computer Vision / image drift monitoring in AERO.

Demonstrates model-agnostic monitoring on image data by:
1. Loading 8x8 grayscale digit images from sklearn.datasets.load_digits.
2. Splitting into reference data (1,000 samples) and unseen control split (797 samples).
3. Training a LogisticRegression classifier on flattened reference digits.
4. Building baseline statistical fingerprint with ImageExtractor (using fixed global scale).
5. Evaluating a FRESH normal control batch (proves 0 false alarms on unseen images).
6. Evaluating all corruptions: Darker, Noisy, Blurry, and Mixed.
7. Printing baseline feature means, drift scores, risk levels, classifier accuracy,
   and PSI feature breakdown tables for all scenarios.
"""
from sklearn.linear_model import LogisticRegression

from data_gen import make_image_data
from extractors import get_extractor
from monitor import build_baseline, check_batch, level, diagnose


def print_psi_table(psi_dict):
    """Print formatted PSI breakdown table."""
    print("  Feature                PSI Value    Drift Severity  Status")
    print("  " + "-" * 60)
    for feat, psi_val in sorted(psi_dict.items(), key=lambda kv: -kv[1]):
        if psi_val > 0.25:
            severity = "Large Drift"
            indicator = "[DRIFTED]"
        elif psi_val > 0.10:
            severity = "Moderate Drift"
            indicator = "[WARNING]"
        else:
            severity = "Stable"
            indicator = "[OK]"
        print(f"  {feat:<22} {psi_val:<12.4f} {severity:<15} {indicator}")
    print("  " + "-" * 60)


def main():
    print("=" * 65)
    print("AERO: Model-Agnostic Image Drift Test (Global Normalization)")
    print("=" * 65)

    # Step 1: Load reference digits and create reference vs unseen control splits
    print("\n[Step 1] Loading digits and creating reference vs unseen control splits...")
    imgs_all, y_all = make_image_data(drifted=False, seed=42)
    X_train, y_train = imgs_all[:1000], y_all[:1000]
    X_control, y_control = imgs_all[1000:], y_all[1000:]
    print(f"  - Reference samples  : {len(X_train)} images of shape {X_train[0].shape}")
    print(f"  - Control samples    : {len(X_control)} unseen images")

    # Step 2: Train LogisticRegression classifier on flattened images
    print("\n[Step 2] Training LogisticRegression on flattened images...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train.reshape(len(X_train), -1), y_train)
    acc_train = float((clf.predict(X_train.reshape(len(X_train), -1)) == y_train).mean())
    print(f"  - Classifier trained successfully. Reference accuracy: {acc_train * 100:.1f}%")

    # Step 3: Extract features and build baseline fingerprint using fixed global scale
    print("\n[Step 3] Building baseline statistical fingerprint with ImageExtractor...")
    extractor = get_extractor("image")
    proba_train = clf.predict_proba(X_train.reshape(len(X_train), -1))
    baseline = build_baseline(X_train, proba_train, y=y_train, extractor=extractor)
    print(f"  - Inferred fixed global max_value: {extractor.max_value} (stored in baseline)")

    print(f"\nBaseline Feature Means ({len(baseline['features'])} features):")
    print("-" * 65)
    for feat in baseline["features"]:
        b = baseline["features"][feat]
        print(f"  - {feat:<20}: mean = {b['mean']:.4f}, std = {b['std']:.4f}")
    print("-" * 65)

    # Step 4: Evaluate all scenarios
    scenarios = [
        ("Normal (Control)", None),
        ("Darker", "dark"),
        ("Noisy", "noise"),
        ("Blurry", "blur"),
        ("Mixed Corruption", "mixed"),
    ]

    results_summary = []

    for name, corr in scenarios:
        print(f"\nEvaluating Scenario: {name.upper()}")
        print("=" * 65)
        if corr is None:
            X_batch, y_batch = X_control, y_control
        else:
            X_batch, y_batch = make_image_data(drifted=True, seed=99, corruption=corr)

        proba_batch = clf.predict_proba(X_batch.reshape(len(X_batch), -1))
        acc_batch = float((clf.predict(X_batch.reshape(len(X_batch), -1)) == y_batch).mean())
        res_batch = check_batch(X_batch, proba_batch, baseline, extractor=extractor)
        diag_text, action, main_cause = diagnose(res_batch)
        risk_lvl = level(res_batch["risk_raw"])

        results_summary.append({
            "scenario": name,
            "drift_score": res_batch["drift_score"],
            "risk_raw": res_batch["risk_raw"],
            "level": risk_lvl,
            "accuracy": acc_batch * 100,
        })

        print(f"Accuracy         : {acc_batch * 100:.1f}%")
        print(f"Drift Score      : {res_batch['drift_score']:.1f} / 100")
        print(f"Overall Risk     : {res_batch['risk_raw']:.1f} / 100")
        print(f"Risk Level       : {risk_lvl}")
        print(f"Primary Cause    : {main_cause.upper()}")
        print(f"Diagnosis        : {diag_text}")
        print(f"Action           : {action}")
        print("\nFeature PSI Breakdown:")
        print_psi_table(res_batch["psi"])

    # Step 5: Overall summary table across all scenarios
    print("\n" + "=" * 65)
    print("OVERALL SUMMARY ACROSS ALL SCENARIOS:")
    print("=" * 65)
    print(f"{'Scenario':<20} {'Accuracy':<10} {'Drift Score':<14} {'Risk Score':<12} {'Risk Level'}")
    print("-" * 65)
    for r in results_summary:
        print(f"{r['scenario']:<20} {r['accuracy']:>6.1f}%    {r['drift_score']:>6.1f} / 100   {r['risk_raw']:>6.1f} / 100  {r['level']}")
    print("-" * 65)
    print("Image drift monitoring test suite completed successfully!\n")


if __name__ == "__main__":
    main()
