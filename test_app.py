"""Automated AppTest suite for AERO multi-modal dashboard.

Tests model cards, status banners (HEALTHY, WATCH, DANGER), deduplicated toasts,
and all scenarios across Structured data, Text, and Images streams.
"""
from streamlit.testing.v1 import AppTest


def run_tests():
    print("=" * 70)
    print("AERO AppTest: Status Banner & Feedback System Verification")
    print("=" * 70)

    caption_expected = "AERO only needs what goes into a model and what comes out. Swap the model and the monitoring keeps working."

    # -------------------------------------------------------------------------
    # 1. Test Structured data
    # -------------------------------------------------------------------------
    print("\n[1] Testing Stream: Structured data")
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception, f"Structured data default exception: {at.exception}"
    assert at.sidebar.selectbox[0].value == "Structured data"
    headers = [h.value for h in at.header]
    assert "Monitoring results" in headers, f"Expected Monitoring results in headers, got {headers}"

    # Verify model card table & caption
    captions = [c.value for c in at.caption]
    assert any(caption_expected in c for c in captions), "Expected model card caption not found in Structured data"
    all_text = " ".join([m.value for m in at.markdown])
    assert "Logistic Regression (scikit-learn)" in all_text, "Connected model not found in table"
    assert "Signals received" in all_text, "Signals received not found in table"
    assert "Replaceable with" in all_text, "Replaceable with not found in table"
    print("  [PASS] Structured data model card table & caption verified.")

    # Default is gradual_drift at batch 20 (DANGER)
    assert len(at.error) > 0, "Expected error banner for default gradual_drift batch 20"
    danger_banner = at.error[0].value
    assert "DRIFT DETECTED, ACTION RECOMMENDED" in danger_banner
    assert "Check complete:" in danger_banner
    assert "5 shifted, 2 showing early warning, 0 stable (7 monitored signals)." in danger_banner
    assert "Most affected:" in danger_banner
    assert "debt_ratio, age, loan_amount_k" in danger_banner
    assert "What it means:" in danger_banner
    assert "Incoming data no longer looks like the data the model was built on. Accuracy may fall." in danger_banner
    assert "Next step:" in danger_banner
    print("  [PASS] Structured data DANGER status banner verified.")

    # Check toast on check finish & deduplication on rerun
    assert len(at.toast) > 0, "Expected toast on check completion"
    assert at.toast[0].value == "Check complete: risk level DANGER"
    at.run()
    assert len(at.toast) == 0, f"Expected no duplicate toast on rerun, got {at.toast}"
    print("  [PASS] Toast fired and deduplicated on rerun (no duplicate on static rerun).")

    # Test gradual_drift at batch 12 (Middle Level: WATCH)
    at.sidebar.slider[0].set_value(12).run()
    assert not at.exception, f"Batch 12 exception: {at.exception}"
    assert len(at.warning) > 0, "Expected warning banner for batch 12 (WATCH)"
    watch_banner = at.warning[0].value
    assert "EARLY WARNING" in watch_banner
    assert "Check complete:" in watch_banner
    assert "0 shifted, 0 showing early warning, 7 stable (7 monitored signals)." in watch_banner
    assert "Most affected:" in watch_banner
    assert "None (all signals stable)" in watch_banner
    assert "What it means:" in watch_banner
    assert "Input signals are stable, but the model's output behavior (confidence and prediction patterns) has started to change. Accuracy may fall if this continues." in watch_banner
    assert "distribution shifts" not in watch_banner
    assert "Next step:" in watch_banner
    print("  [PASS] Structured data middle level (WATCH) status banner verified.")

    # Test control (HEALTHY)
    at.sidebar.selectbox[1].select("control").run()
    assert not at.exception, f"Control scenario exception: {at.exception}"
    assert len(at.success) > 0, "Expected success banner for control scenario (HEALTHY)"
    healthy_banner = at.success[0].value
    assert "ALL CLEAR" in healthy_banner
    assert "Check complete:" in healthy_banner
    assert "all 7 monitored signals are stable." in healthy_banner
    assert "Most affected:" in healthy_banner
    assert "None (all signals stable)" in healthy_banner
    assert "What it means:" in healthy_banner
    assert "Incoming data matches the baseline." in healthy_banner
    assert "Next step:" in healthy_banner
    print("  [PASS] Structured data HEALTHY status banner verified.")

    # Test quality_break (HIGH / DANGER at batch 20)
    at.sidebar.selectbox[1].select("quality_break").run()
    at.sidebar.slider[0].set_value(20).run()
    assert not at.exception, f"quality_break exception: {at.exception}"
    assert len(at.error) > 0, "Expected error banner for quality_break at batch 20 (HIGH)"
    assert "DRIFT DETECTED, ACTION RECOMMENDED" in at.error[0].value
    print("  [PASS] Structured data 'quality_break' loaded cleanly.")

    # -------------------------------------------------------------------------
    # 2. Test Text
    # -------------------------------------------------------------------------
    print("\n[2] Testing Stream: Text")
    at.sidebar.selectbox[0].select("Text").run()
    assert not at.exception, f"Text stream exception: {at.exception}"
    assert at.sidebar.selectbox[0].value == "Text"

    # Default scenario: Normal reviews (HEALTHY)
    assert len(at.success) > 0, "Expected success banner for Normal reviews"
    text_healthy_banner = at.success[0].value
    assert "ALL CLEAR" in text_healthy_banner
    assert "Check complete:" in text_healthy_banner
    assert "all 7 monitored signals are stable." in text_healthy_banner
    assert "Most affected:" in text_healthy_banner
    assert "None (all signals stable)" in text_healthy_banner
    assert "What it means:" in text_healthy_banner
    assert "Incoming data matches the baseline." in text_healthy_banner
    assert "Next step:" in text_healthy_banner
    assert "No action needed. Continue monitoring." in text_healthy_banner
    print("  [PASS] Text 'Normal reviews' HEALTHY status banner verified.")

    # Drifted reviews (DANGER)
    at.sidebar.selectbox[1].select("Drifted reviews").run()
    assert not at.exception, f"Text Drifted exception: {at.exception}"
    assert len(at.error) > 0, "Expected error banner for Drifted reviews"
    text_danger_banner = at.error[0].value
    assert "DRIFT DETECTED, ACTION RECOMMENDED" in text_danger_banner
    assert "Check complete:" in text_danger_banner
    assert "7 shifted, 0 showing early warning, 0 stable (7 monitored signals)." in text_danger_banner
    assert "What it means:" in text_danger_banner
    assert "Incoming data no longer looks like the data the model was built on. Accuracy may fall." in text_danger_banner
    print("  [PASS] Text 'Drifted reviews' DANGER status banner verified.")

    # -------------------------------------------------------------------------
    # 3. Test Images
    # -------------------------------------------------------------------------
    print("\n[3] Testing Stream: Images")
    at.sidebar.selectbox[0].select("Images").run()
    assert not at.exception, f"Images stream exception: {at.exception}"
    assert at.sidebar.selectbox[0].value == "Images"

    # Default scenario: Normal (HEALTHY)
    assert len(at.success) > 0, "Expected success banner for Normal digits"
    img_healthy_banner = at.success[0].value
    assert "ALL CLEAR" in img_healthy_banner
    assert "Check complete:" in img_healthy_banner
    assert "all 6 monitored signals are stable." in img_healthy_banner
    assert "Most affected:" in img_healthy_banner
    assert "None (all signals stable)" in img_healthy_banner
    assert "What it means:" in img_healthy_banner
    assert "Incoming data matches the baseline." in img_healthy_banner
    assert "Next step:" in img_healthy_banner
    assert "No action needed. Continue monitoring." in img_healthy_banner
    print("  [PASS] Images 'Normal' HEALTHY status banner verified.")

    for sc in ["Darker", "Noisy", "Blurry", "Mixed corruption"]:
        at.sidebar.selectbox[1].select(sc).run()
        assert not at.exception, f"Images {sc} exception: {at.exception}"
        assert len(at.error) > 0, f"Expected error banner for Images {sc}"
        assert "DRIFT DETECTED, ACTION RECOMMENDED" in at.error[0].value
        if sc == "Mixed corruption":
            mixed_banner = at.error[0].value
            assert "6 shifted, 0 showing early warning, 0 stable (6 monitored signals)." in mixed_banner
            assert "bright_ratio, brightness, contrast" in mixed_banner
        print(f"  [PASS] Images '{sc}' DANGER status banner verified.")

    print("\n" + "=" * 70)
    print("ALL APPTEST CHECKS PASSED: STATUS BANNERS & FEEDBACK VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
