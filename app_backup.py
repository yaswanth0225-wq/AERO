"""Step 3: the dashboard.  Run with:  streamlit run app.py"""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitor import WARN, HIGH, DANGER

st.set_page_config(page_title="AERO - AI Model Reliability Forecaster", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .aero-header {
        text-align: center;
        margin-top: -30px;
        margin-bottom: 24px;
        padding-bottom: 16px;
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
    .aero-center-caption {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    </style>
    <div class="aero-header">
        <h1 class="aero-center-title"><span class="aero-title-brand">AERO</span> - AI Model Reliability Forecaster</h1>
        <p class="aero-center-caption">Warns BEFORE accuracy falls, without needing the true labels. Only statistical summaries are kept; raw records are never stored.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    res = pd.read_csv("results/results.csv")
    summary = json.load(open("results/summary.json"))
except FileNotFoundError:
    st.error("Run `python train.py` and then `python simulate.py` first.")
    st.stop()

COLORS = {"HEALTHY": "#2e9e5b", "WATCH": "#e0b000", "HIGH": "#e8742a", "DANGER": "#d63a3a"}
scenario = st.sidebar.selectbox("Scenario", ["gradual_drift", "quality_break", "control"],
                                format_func=lambda s: {"gradual_drift": "Gradual real-world drift",
                                                       "quality_break": "Broken data feed",
                                                       "control": "Control (no drift)"}[s])
df = res[res.scenario == scenario].reset_index(drop=True)
now = st.sidebar.slider("Time (batch number)", 0, int(df.batch.max()), 20)
show_truth = st.sidebar.checkbox("Show hidden true accuracy (backtest only)", value=True)
row = df.iloc[now]

# ---- top row: gauge + alert ----
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

# ---- risk over time ----
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

# ---- which features drifted ----
st.subheader("Which features shifted most (PSI)")
psi = {c[4:]: row[c] for c in df.columns if c.startswith("psi_")}
p = pd.Series(psi).sort_values()
bar = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h",
                       marker_color=["#d63a3a" if v > 0.25 else "#e0b000" if v > 0.1 else "#2e9e5b" for v in p.values]))
bar.update_layout(height=300, xaxis_title="PSI  (<0.1 stable, 0.1-0.25 moderate, >0.25 large)", margin=dict(t=10))
st.plotly_chart(bar, width="stretch")

# ---- backtest proof ----
st.subheader("Backtest proof (10 simulated runs per scenario)")
g, q, ctl = summary["scenarios"]["gradual_drift"], summary["scenarios"]["quality_break"], summary["scenarios"]["control"]
k1, k2, k3 = st.columns(3)
k1.metric("Warned before accuracy failure", f"{g['warned_before_failure']} / {g['runs_with_real_failure']} runs")
k2.metric("Median lead time", f"{g['median_lead_time_batches']:.1f} batches" if g["median_lead_time_batches"] else "n/a")
k3.metric("False alarms on no-drift control", f"{ctl['false_alarm_rate']*100:.0f}%")
st.caption(f"Data-outage scenario: {q['alerts_raised']}/{q['runs']} runs flagged and diagnosed as a data-quality issue. "
           f"{q['note']}")
