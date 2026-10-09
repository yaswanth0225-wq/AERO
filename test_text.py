"""Test script for NLP / text drift monitoring in AERO.

Demonstrates model-agnostic monitoring by:
1. Generating ~500 reference product reviews and training a simple text classifier
   (scikit-learn TfidfVectorizer + LogisticRegression).
2. Building a reference baseline fingerprint using TextExtractor from extractors.py.
3. Generating ~500 drifted texts (longer, uppercase, numbers, gibberish).
4. Running check_batch to compute distribution drift (PSI) and risk scores.
5. Printing which features drifted most and the final drift score.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_gen import make_text_data
from extractors import get_extractor
from monitor import build_baseline, check_batch, level, diagnose


def main():
    print("=" * 65)
    print("AERO: Model-Agnostic Text Drift Test")
    print("=" * 65)

    # Step 1: Generate reference text dataset (~500 short normal reviews)
    print("\n[Step 1] Generating reference data (~500 reviews)...")
    texts_ref, y_ref = make_text_data(n=500, seed=42, drifted=False)
    print(f"  - Sample reference text: '{texts_ref[0]}'")

    # Step 2: Train a simple text classifier (TfidfVectorizer + LogisticRegression)
    print("\n[Step 2] Training scikit-learn text classifier (TF-IDF + LogisticRegression)...")
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("lr", LogisticRegression(random_state=42))
    ])
    clf.fit(texts_ref, y_ref)
    proba_ref = clf.predict_proba(texts_ref)
    acc_ref = float((clf.predict(texts_ref) == y_ref).mean())
    print(f"  - Classifier trained successfully. Reference accuracy: {acc_ref * 100:.1f}%")

    # Step 3: Extract features and build baseline fingerprint
    print("\n[Step 3] Building baseline statistical fingerprint with TextExtractor...")
    extractor = get_extractor("text")
    baseline = build_baseline(texts_ref, proba_ref, y=y_ref, extractor=extractor)
    print(f"  - Extracted {len(baseline['features'])} text features:")
    for feat in baseline["features"]:
        b = baseline["features"][feat]
        print(f"    - {feat:<20}: mean = {b['mean']:.3f}, std = {b['std']:.3f}")

    # Step 4: Control check on a FRESH unseen normal batch (different seed, not training data)
    print("\n[Step 4] Evaluating FRESH normal control batch (different seed=123, unseen data)...")
    texts_control, y_control = make_text_data(n=500, seed=123, drifted=False)
    proba_control = clf.predict_proba(texts_control)
    res_control = check_batch(texts_control, proba_control, baseline, extractor=extractor)
    print(f"  - Control drift score: {res_control['drift_score']:.1f} / 100 (Status: {level(res_control['risk_raw'])})")

    # Step 5: Generate drifted text data (~500 long, all-caps, numeric, gibberish)
    print("\n[Step 5] Generating drifted text data (~500 samples)...")
    texts_drifted, y_drifted = make_text_data(n=500, seed=99, drifted=True)
    print(f"  - Sample drifted text: '{texts_drifted[0][:90]}...'")

    # Step 6: Run drift check on drifted data
    print("\n[Step 6] Running drift check on drifted batch...")
    proba_drifted = clf.predict_proba(texts_drifted)
    res_drift = check_batch(texts_drifted, proba_drifted, baseline, extractor=extractor)
    diag_text, action, main_cause = diagnose(res_drift)

    # Step 7: Print results, drifted features, and drift score
    print("\n" + "=" * 65)
    print("RESULTS:")
    print("=" * 65)
    print(f"Drift Score   : {res_drift['drift_score']:.1f} / 100")
    print(f"Overall Risk  : {res_drift['risk_raw']:.1f} / 100")
    print(f"Risk Level    : {level(res_drift['risk_raw'])}")
    print(f"Primary Cause : {main_cause.upper()}")
    print(f"Diagnosis     : {diag_text}")
    print(f"Action        : {action}")

    print("\nWhich Features Drifted (sorted by Population Stability Index - PSI):")
    print("-" * 65)
    print(f"{'Feature':<22} {'PSI Value':<12} {'Drift Severity':<15} {'Status'}")
    print("-" * 65)
    for feat, psi_val in sorted(res_drift["psi"].items(), key=lambda kv: -kv[1]):
        if psi_val > 0.25:
            severity = "Large Drift"
            indicator = "[DRIFTED]"
        elif psi_val > 0.10:
            severity = "Moderate Drift"
            indicator = "[WARNING]"
        else:
            severity = "Stable"
            indicator = "[OK]"
        print(f"{feat:<22} {psi_val:<12.4f} {severity:<15} {indicator}")
    print("-" * 65)
    print("\nDrift detection test completed successfully!")


if __name__ == "__main__":
    main()
