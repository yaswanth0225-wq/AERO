"""AERO - AI Model Reliability Forecaster Dashboard.

Supports model-agnostic drift monitoring across Tabular, Text, and Image data.
Run with:  streamlit run app.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitor import WARN, HIGH, DANGER, level, diagnose

st.set_page_config(page_title="AERO - AI Model Reliability Forecaster", page_icon="🛡️", layout="wide")

# Custom header styling
st.markdown(
    """
    <style>
    .aero-header {
        text-align: center;
        margin-top: -30px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .aero-center-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }
    .aero-title-brand {
        color: #2563EB;
        font-weight: 900;
        letter-spacing: 1.5px;
    }
    .aero-tagline {
        font-size: 1.15rem;
        margin-top: 8px;
        margin-bottom: 0;
    }
    </style>
    <div class="aero-header">
        <h1 class="aero-center-title"><span class="aero-title-brand">AERO</span> - AI Model Reliability Forecaster</h1>
        <p class="aero-tagline"><strong>Know your AI model is about to fail before it does.</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How it works"):
    st.markdown(
        """
        - **Drift:** checks whether incoming data looks different from the data the model learned from.
        - **Data quality:** checks for missing, broken or impossible values.
        - **Model behaviour:** checks whether the model is becoming less confident or answering differently.

        These are combined into one 0-100 risk score, and a trend forecast estimates how many batches remain before it becomes unsafe. No true labels are needed.
        """
    )

COLORS = {"HEALTHY": "#2e9e5b", "WATCH": "#e0b000", "HIGH": "#e8742a", "DANGER": "#d63a3a"}

# =============================================================================
# CACHED RESOURCE & DATA LOADERS (Streamlit Cloud Performance Optimization)
# =============================================================================

@st.cache_resource
def get_text_model_and_baseline():
    """Train demo text model and build reference baseline once."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from data_gen import make_text_data
    from extractors import get_extractor
    from monitor import build_baseline

    texts_ref, y_ref = make_text_data(n=500, seed=42, drifted=False)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("lr", LogisticRegression(random_state=42))
    ])
    clf.fit(texts_ref, y_ref)
    proba_ref = clf.predict_proba(texts_ref)
    extractor = get_extractor("text")
    baseline = build_baseline(texts_ref, proba_ref, y=y_ref, extractor=extractor)
    return clf, baseline, extractor


@st.cache_data
def get_text_dataset(scenario_name: str):
    """Generate text dataset for selected scenario."""
    from data_gen import make_text_data
    if scenario_name == "Normal reviews":
        return make_text_data(n=500, seed=123, drifted=False)
    return make_text_data(n=500, seed=99, drifted=True)


@st.cache_resource
def get_image_model_and_baseline():
    """Train demo image classifier and build reference baseline once."""
    from sklearn.linear_model import LogisticRegression
    from data_gen import make_image_data
    from extractors import get_extractor
    from monitor import build_baseline

    imgs_all, y_all = make_image_data(drifted=False, seed=42)
    X_train, y_train = imgs_all[:1000], y_all[:1000]
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train.reshape(len(X_train), -1), y_train)
    proba_train = clf.predict_proba(X_train.reshape(len(X_train), -1))
    extractor = get_extractor("image")
    baseline = build_baseline(X_train, proba_train, y=y_train, extractor=extractor)
    return clf, baseline, extractor


@st.cache_data
def get_image_dataset(scenario_name: str):
    """Generate image dataset for selected scenario."""
    from data_gen import make_image_data
    corr_map = {
        "Normal": None,
        "Darker": "dark",
        "Noisy": "noise",
        "Blurry": "blur",
        "Mixed corruption": "mixed",
    }
    corr = corr_map.get(scenario_name)
    if corr is None:
        imgs_all, y_all = make_image_data(drifted=False, seed=42)
        return imgs_all[1000:], y_all[1000:]
    return make_image_data(drifted=True, seed=99, corruption=corr)


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

st.sidebar.markdown("### Monitor Configuration")
data_type = st.sidebar.selectbox("Monitored data stream", ["Structured data", "Text", "Images"])
st.sidebar.caption("AERO works with any model. These are three demo integrations.")

TOAST_ICONS = {
    "HEALTHY": "✅",
    "WATCH": "⚠️",
    "MODERATE": "⚠️",
    "HIGH": "⚠️",
    "DANGER": "🚨",
}


def render_status_banner(res_level: str, psi_dict: dict, action_text: str, primary_cause: str = "NONE"):
    """Render top status banner based on monitoring risk level and primary cause."""
    n_total = len(psi_dict)
    shifted_feats = [f for f, p in psi_dict.items() if p > 0.25]
    warning_feats = [f for f, p in psi_dict.items() if 0.10 < p <= 0.25]
    stable_feats = [f for f, p in psi_dict.items() if p <= 0.10]

    x_shifted = len(shifted_feats)
    y_warning = len(warning_feats)
    z_stable = len(stable_feats)
    total_shifting = x_shifted + y_warning

    # Affected features are strictly those with DRIFTED (>0.25) or WARNING (>0.10) status
    affected_feats = [f for f, p in psi_dict.items() if p > 0.10]
    if affected_feats:
        sorted_affected = sorted(affected_feats, key=lambda f: psi_dict[f], reverse=True)
        most_affected = ", ".join(sorted_affected[:3])
    else:
        most_affected = "None (all signals stable)"

    cause_str = (primary_cause or "").strip().lower()

    if res_level == "HEALTHY":
        title = "ALL CLEAR"
        check_line = f"all {n_total} monitored signals are stable."
        meaning_line = "Incoming data matches the baseline."
        render_fn = st.success
    elif total_shifting == 0:
        # Shifted + early warning count is 0, but risk is above HEALTHY
        if res_level in ("WATCH", "MODERATE"):
            title = "EARLY WARNING"
            render_fn = st.warning
        else:
            title = "DRIFT DETECTED, ACTION RECOMMENDED"
            render_fn = st.error

        check_line = f"{x_shifted} shifted, {y_warning} showing early warning, {z_stable} stable ({n_total} monitored signals)."
        if cause_str == "quality":
            meaning_line = (
                "Input signals are stable, but data quality has started to degrade "
                "(for example missing or invalid values). Accuracy may fall if this continues."
            )
        else:
            meaning_line = (
                "Input signals are stable, but the model's output behavior "
                "(confidence and prediction patterns) has started to change. Accuracy may fall if this continues."
            )
    elif res_level in ("WATCH", "MODERATE"):
        title = "EARLY WARNING"
        check_line = f"{x_shifted} shifted, {y_warning} showing early warning, {z_stable} stable ({n_total} monitored signals)."
        meaning_line = "Incoming data shows early distribution shifts from the baseline. Accuracy may fall if drift continues."
        render_fn = st.warning
    else:  # HIGH, DANGER with shifted or early-warning signals
        title = "DRIFT DETECTED, ACTION RECOMMENDED"
        check_line = f"{x_shifted} shifted, {y_warning} showing early warning, {z_stable} stable ({n_total} monitored signals)."
        meaning_line = "Incoming data no longer looks like the data the model was built on. Accuracy may fall."
        render_fn = st.error

    clean_action = action_text.strip()

    banner_md = (
        f"**{title}**  \n"
        f"**Check complete:** {check_line}  \n"
        f"**Most affected:** {most_affected}  \n"
        f"**What it means:** {meaning_line}  \n"
        f"**Next step:** {clean_action}"
    )
    render_fn(banner_md)
    return banner_md


def trigger_check_toast(stream: str, scenario: str, res_level: str, extra_key: str = None):
    """Trigger a toast once when a monitoring check finishes for a stream/scenario."""
    toast_state_key = (stream, scenario, extra_key) if extra_key is not None else (stream, scenario)
    if st.session_state.get("last_toast_key") != toast_state_key:
        st.session_state["last_toast_key"] = toast_state_key
        st.session_state["last_shown"] = (stream, scenario)
        st.session_state["last_toast_shown"] = (stream, scenario)
        icon = TOAST_ICONS.get(res_level, "ℹ️")
        st.toast(f"Check complete: risk level {res_level}", icon=icon)


# =============================================================================
# STREAM 1: STRUCTURED DATA
# =============================================================================
if data_type == "Structured data":
    st.header("Monitoring results")
    with st.expander("Connected demo model", expanded=False):
        st.markdown(
            """
| Property | Specification |
| :--- | :--- |
| **Connected model** | Logistic Regression (scikit-learn) |
| **Task** | Customer financial risk classification |
| **Signals received** | Input features, predicted probabilities |
| **Signals not needed** | Model weights, source code, training data |
| **Integration** | Read-only: observes the model, never modifies it |
| **Replaceable with** | XGBoost, neural networks, hosted model APIs |
"""
        )
        st.caption("AERO only needs what goes into a model and what comes out. Swap the model and the monitoring keeps working.")

    try:
        res = pd.read_csv("results/results.csv")
        summary = json.load(open("results/summary.json"))
    except FileNotFoundError:
        st.error("Run `python train.py` and then `python simulate.py` first.")
        st.stop()

    scenario = st.sidebar.selectbox(
        "Scenario",
        ["gradual_drift", "quality_break", "control"],
        format_func=lambda s: {
            "gradual_drift": "Gradual real-world drift",
            "quality_break": "Broken data feed",
            "control": "Control (no drift)",
        }[s],
    )

    df = res[res.scenario == scenario].reset_index(drop=True)
    now = st.sidebar.slider("Time (batch number)", 0, int(df.batch.max()), 20)
    show_truth = st.sidebar.checkbox("Show hidden true accuracy (backtest only)", value=True)
    row = df.iloc[now]

    psi = {c[4:]: row[c] for c in df.columns if c.startswith("psi_")}
    primary_cause = getattr(row, "main_cause", "NONE")
    render_status_banner(row.level, psi, row.action, primary_cause=primary_cause)
    trigger_check_toast("Structured data", scenario, row.level, extra_key=str(now))

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Drift score", f"{row.drift_score:.0f} / 100")
    m2.metric("Overall risk", f"{row.risk:.0f} / 100")
    m3.metric("Risk level", row.level)

    # Top row: gauge + status alert
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=float(row.risk), title={"text": "Failure-risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[row.level]},
                   "steps": [{"range": [0, WARN], "color": "#e6f4ec"},
                             {"range": [WARN, HIGH], "color": "#fbf3d0"},
                             {"range": [HIGH, DANGER], "color": "#fbe0cf"},
                             {"range": [DANGER, 100], "color": "#f6d3d3"}]}))
        fig.update_layout(height=280, margin=dict(t=60, b=0, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(f"### Status: :{ {'HEALTHY':'green','WATCH':'orange','HIGH':'orange','DANGER':'red'}[row.level] }[{row.level}]")
        if pd.notna(row.eta_batches):
            st.warning(f"⏱️ Forecast: risk reaches the DANGER level in about **{row.eta_batches:.0f} batches**")
        elif row.level == "HEALTHY":
            st.success("No rising trend detected.")
        st.write(f"**Diagnosis:** {row.diagnosis}")
        st.write(f"**Suggested action:** {row.action}")
        a, b, c = st.columns(3)
        a.metric("Drift score", f"{row.drift_score:.0f}")
        b.metric("Data-quality score", f"{row.quality_score:.0f}")
        c.metric("Behaviour score", f"{row.behavior_score:.0f}")

        if st.button("Dispatch Mitigation"):
            st.toast("Action Dispatched: Pipeline alert sent to MLOps team!", icon="🚀")
            st.success("Mitigation logged and human review requested.")

    # Risk over time chart
    st.subheader("Risk over time")
    t = go.Figure()
    t.add_trace(go.Scatter(x=df.batch, y=df.risk, name="Risk score", line=dict(width=4, color="#3b5bdb")))
    t.add_trace(go.Scatter(x=df.batch, y=df.drift_score, name="Drift", line=dict(dash="dot")))
    t.add_trace(go.Scatter(x=df.batch, y=df.quality_score, name="Data quality", line=dict(dash="dot")))
    t.add_trace(go.Scatter(x=df.batch, y=df.behavior_score, name="Behaviour", line=dict(dash="dot")))
    if show_truth:
        t.add_trace(go.Scatter(x=df.batch, y=df.acc_roll * 100, name="TRUE accuracy % (hidden from monitor)",
                               yaxis="y2", line=dict(color="black", width=2)))
    t.add_hline(y=WARN, line_dash="dash", line_color="#e0b000")
    t.add_vline(x=now, line_color="gray")
    t.update_layout(height=380, yaxis=dict(title="Score (0-100)", range=[0, 105]),
                    yaxis2=dict(title="Accuracy %", overlaying="y", side="right", range=[50, 90]),
                    legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(t, width="stretch")

    # Features shifted most
    st.subheader("Which features shifted most (PSI)")
    psi = {c[4:]: row[c] for c in df.columns if c.startswith("psi_")}
    p = pd.Series(psi).sort_values()
    bar = go.Figure(go.Bar(
        x=p.values, y=p.index, orientation="h",
        marker_color=["#d63a3a" if v > 0.25 else "#e0b000" if v > 0.1 else "#2e9e5b" for v in p.values]
    ))
    bar.update_layout(height=300, xaxis_title="PSI  (<0.1 stable, 0.1-0.25 moderate, >0.25 large)", margin=dict(t=10))
    st.plotly_chart(bar, width="stretch")

    # Backtest proof
    st.subheader("Backtest proof (10 simulated runs per scenario)")
    g, q, ctl = summary["scenarios"]["gradual_drift"], summary["scenarios"]["quality_break"], summary["scenarios"]["control"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Warned before accuracy failure", f"{g['warned_before_failure']} / {g['runs_with_real_failure']} runs")
    k2.metric("Median lead time", f"{g['median_lead_time_batches']:.1f} batches" if g["median_lead_time_batches"] else "n/a")
    k3.metric("False alarms on no-drift control", f"{ctl['false_alarm_rate']*100:.0f}%")
    st.caption(f"Data-outage scenario: {q['alerts_raised']}/{q['runs']} runs flagged and diagnosed as a data-quality issue. {q['note']}")


# =============================================================================
# STREAM 2: TEXT
# =============================================================================
elif data_type == "Text":
    st.header("Monitoring results")
    with st.expander("Connected demo model", expanded=False):
        st.markdown(
            """
| Property | Specification |
| :--- | :--- |
| **Connected model** | TF-IDF + Logistic Regression (scikit-learn) |
| **Task** | Review sentiment classification |
| **Signals received** | Input text, predicted probabilities |
| **Signals not needed** | Model weights, source code, training data |
| **Integration** | Read-only: observes the model, never modifies it |
| **Replaceable with** | BERT, transformers, LLM APIs |
"""
        )
        st.caption("AERO only needs what goes into a model and what comes out. Swap the model and the monitoring keeps working.")

    scenario = st.sidebar.selectbox("Scenario", ["Normal reviews", "Drifted reviews"])

    # Load model and datasets
    from monitor import check_batch
    clf, baseline, extractor = get_text_model_and_baseline()
    normal_texts, y_normal = get_text_dataset("Normal reviews")
    scenario_texts, y_scenario = get_text_dataset(scenario)

    proba_normal = clf.predict_proba(normal_texts)
    acc_normal = float((clf.predict(normal_texts) == y_normal).mean())

    proba_scenario = clf.predict_proba(scenario_texts)
    acc_scenario = float((clf.predict(scenario_texts) == y_scenario).mean())

    res = check_batch(scenario_texts, proba_scenario, baseline, extractor=extractor)
    res_level = level(res["risk_raw"])
    diag_text, action_text, cause = diagnose(res)

    render_status_banner(res_level, res["psi"], action_text, primary_cause=cause)
    trigger_check_toast("Text", scenario, res_level)

    # Metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Drift score", f"{res['drift_score']:.1f} / 100")
    m2.metric("Overall risk", f"{res['risk_raw']:.1f} / 100")
    m3.metric("Risk level", res_level)

    # Accuracy comparison
    a1, a2 = st.columns(2)
    a1.metric("Normal batch accuracy", f"{acc_normal * 100:.1f}%")
    drop = (acc_scenario - acc_normal) * 100
    a2.metric("Scenario batch accuracy", f"{acc_scenario * 100:.1f}%", delta=f"{drop:+.1f}%")

    # Diagnosis and action
    st.markdown(f"### Status: :{ {'HEALTHY':'green','WATCH':'orange','HIGH':'orange','DANGER':'red'}[res_level] }[{res_level}]")
    st.write(f"**Diagnosis:** {diag_text}")
    st.write(f"**Suggested action:** {action_text}")

    # Feature drift breakdown
    st.subheader("Which features shifted most (PSI)")
    p = pd.Series(res["psi"]).sort_values()
    bar = go.Figure(go.Bar(
        x=p.values, y=p.index, orientation="h",
        marker_color=["#d63a3a" if v > 0.25 else "#e0b000" if v > 0.1 else "#2e9e5b" for v in p.values]
    ))
    bar.update_layout(height=320, xaxis_title="PSI  (<0.1 stable, 0.1-0.25 moderate, >0.25 large)", margin=dict(t=10))
    st.plotly_chart(bar, width="stretch")

    # Sample text inspection table
    st.subheader("Sample Text Inspection (3 Normal vs 3 Scenario Reviews)")
    sample_df = pd.DataFrame({
        "Sample #": [1, 2, 3],
        "Normal Review (Reference Baseline)": normal_texts[:3],
        f"Scenario Review ({scenario})": scenario_texts[:3]
    })
    st.dataframe(sample_df, width="stretch", hide_index=True)


# =============================================================================
# STREAM 3: IMAGES
# =============================================================================
elif data_type == "Images":
    st.header("Monitoring results")
    with st.expander("Connected demo model", expanded=False):
        st.markdown(
            """
| Property | Specification |
| :--- | :--- |
| **Connected model** | Logistic Regression on pixel values (scikit-learn) |
| **Task** | Handwritten digit recognition |
| **Signals received** | Input images, predicted probabilities |
| **Signals not needed** | Model weights, source code, training data |
| **Integration** | Read-only: observes the model, never modifies it |
| **Replaceable with** | CNNs, Vision Transformers, hosted vision APIs |
"""
        )
        st.caption("AERO only needs what goes into a model and what comes out. Swap the model and the monitoring keeps working.")

    scenario = st.sidebar.selectbox("Scenario", ["Normal", "Darker", "Noisy", "Blurry", "Mixed corruption"])

    from monitor import check_batch
    clf, baseline, extractor = get_image_model_and_baseline()
    normal_imgs, y_normal = get_image_dataset("Normal")
    scenario_imgs, y_scenario = get_image_dataset(scenario)

    proba_normal = clf.predict_proba(normal_imgs.reshape(len(normal_imgs), -1))
    acc_normal = float((clf.predict(normal_imgs.reshape(len(normal_imgs), -1)) == y_normal).mean())

    proba_scenario = clf.predict_proba(scenario_imgs.reshape(len(scenario_imgs), -1))
    acc_scenario = float((clf.predict(scenario_imgs.reshape(len(scenario_imgs), -1)) == y_scenario).mean())

    res = check_batch(scenario_imgs, proba_scenario, baseline, extractor=extractor)
    res_level = level(res["risk_raw"])
    diag_text, action_text, cause = diagnose(res)

    render_status_banner(res_level, res["psi"], action_text, primary_cause=cause)
    trigger_check_toast("Images", scenario, res_level)

    # Metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Drift score", f"{res['drift_score']:.1f} / 100")
    m2.metric("Overall risk", f"{res['risk_raw']:.1f} / 100")
    m3.metric("Risk level", res_level)

    # Accuracy comparison
    a1, a2 = st.columns(2)
    a1.metric("Normal batch accuracy", f"{acc_normal * 100:.1f}%")
    drop = (acc_scenario - acc_normal) * 100
    a2.metric("Scenario batch accuracy", f"{acc_scenario * 100:.1f}%", delta=f"{drop:+.1f}%")

    # Diagnosis and action
    st.markdown(f"### Status: :{ {'HEALTHY':'green','WATCH':'orange','HIGH':'orange','DANGER':'red'}[res_level] }[{res_level}]")
    st.write(f"**Diagnosis:** {diag_text}")
    st.write(f"**Suggested action:** {action_text}")

    # Feature drift breakdown
    st.subheader("Which features shifted most (PSI)")
    p = pd.Series(res["psi"]).sort_values()
    bar = go.Figure(go.Bar(
        x=p.values, y=p.index, orientation="h",
        marker_color=["#d63a3a" if v > 0.25 else "#e0b000" if v > 0.1 else "#2e9e5b" for v in p.values]
    ))
    bar.update_layout(height=300, xaxis_title="PSI  (<0.1 stable, 0.1-0.25 moderate, >0.25 large)", margin=dict(t=10))
    st.plotly_chart(bar, width="stretch")

    # Visual sample inspection (8 normal vs 8 scenario images)
    st.subheader(f"Visual Sample Inspection (8 Normal vs 8 {scenario} Digits)")
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
    for i in range(8):
        axes[0, i].imshow(normal_imgs[i], cmap="gray_r")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Normal", fontsize=10, fontweight="bold", loc="left")
        axes[1, i].imshow(scenario_imgs[i], cmap="gray_r")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title(scenario, fontsize=10, fontweight="bold", loc="left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption("Prototype built on synthetic data. Monitor stores only statistical summaries, never raw records. Supports SDG 9, 12 and 16.")
