# AERO: the complete beginner's guide (with all the code)

This guide explains the whole project in simple language, then shows every file's code with a plain-English explanation. You do not need to know coding. You need to understand **what each part does and why**, so you can explain it to judges and ask Antigravity for changes.

---

## 1. The big picture (read this first)

### The one-line idea
An AI model is like a student who studied old chapters. When the real world changes, the exam asks about things the student never studied. We can't mark the exam (no answer key yet), but we can **spot warning signs** early. This project is the warning system.

### The smoke-alarm comparison
A smoke alarm does not wait for the house to burn down. It detects smoke, which appears **before** the fire is big. AERO detects three kinds of "smoke" **before** the model's accuracy collapses:

| Smoke signal | Plain meaning | Where it lives in code |
|---|---|---|
| **Drift** | The incoming data looks different from the data the model learned from | `monitor.py` (PSI) |
| **Data quality** | The data is getting broken (blank values, impossible numbers) | `monitor.py` |
| **Model behaviour** | The model is acting strangely (less confident, different answer mix) | `monitor.py` |

### The two "models" in this project (do not mix them up)
1. **The target model**: a normal ML model (Random Forest) that predicts whether a loan customer will default. It is just a **test subject**. Built in `train.py`.
2. **The monitor**: your actual project. It watches the target model and warns when trouble is coming. Lives in `monitor.py`. It is **not** a machine-learning model. It uses statistics and simple maths.

### How the data flows

```
data_gen.py  --makes fake customer data-->  train.py  --> saves: model + "baseline fingerprint"
                                                                    |
simulate.py  --sends 40 batches of data over "time"---------------->|
     |                                                              v
     |        for each batch:  target model predicts  -->  monitor.py scores it
     |                                                     (drift, quality, behaviour -> risk)
     |        ALSO (for the test only): checks the true accuracy
     v
results/ (CSV + JSON files)  -->  app.py (the dashboard you show the judges)
```

### The most important rule in the whole project
**The monitor never sees the true answers (labels).** In real life those arrive weeks later, which is the whole problem. In our test we *do* know the true answers, but only the test script `simulate.py` uses them, to check whether the warning really came early. Judges may ask about this. The answer is: "The monitor code has no access to labels; the labels are used only afterwards to grade it."

### What are "batches"?
Real systems receive data continuously. We group it into **batches** (here, 500 customers per batch) and treat each batch as one "tick" of time, like one day. 40 batches = 40 days.

---

## 2. Setting it up

1. Unzip `driftguard.zip`. You get a folder `driftguard/`.
2. Open that folder in Antigravity.
3. Open a terminal inside Antigravity and run, one at a time:

```
pip install -r requirements.txt
python train.py
python simulate.py
streamlit run app.py
```

What each line does:

| Command | What happens | Time |
|---|---|---|
| `pip install -r requirements.txt` | Installs the libraries the code needs | 1-3 min |
| `python train.py` | Trains the target model and saves its baseline | a few seconds |
| `python simulate.py` | Runs the whole experiment: 3 scenarios x 10 repeats | about 1-2 min |
| `streamlit run app.py` | Opens the dashboard in your browser | instant |

If the terminal confuses you, tell the Antigravity agent: *"Install the requirements, run train.py, then simulate.py, then start the Streamlit app."*

The libraries (in `requirements.txt`): `pandas` (tables), `numpy` (maths), `scikit-learn` (the Random Forest), `scipy`, `joblib` (saving the model), `streamlit` (the web dashboard), `plotly` (charts).

---

## 3. File 1: `data_gen.py` (makes the practice data)

**Job:** create fake loan customers. Each customer has 7 facts (age, income, debt ratio, etc.) and a true answer: defaulted (1) or not (0).

**Why fake data?** So we control exactly *when* and *how* the world changes. That lets us measure honestly whether our warning came early. (If you have time, you can later swap in a real dataset.)

```python
"""Synthetic credit-risk data generator.

`drift` goes from 0.0 (same world as training) to 1.0 (very different world).
`missing_rate` = fraction of 'income' values that go missing (pipeline break).
`unit_error_rate` = fraction of rows from a broken upstream feed (a feed outage zero-fills debt ratio, utilisation and late payments).
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "income_k", "debt_ratio", "credit_utilization",
            "late_payments", "account_age_years", "loan_amount_k"]


def make_data(n, seed, drift=0.0, missing_rate=0.0, unit_error_rate=0.0):
    rng = np.random.default_rng(seed)
    d = float(drift)

    age = np.clip(rng.normal(42 - 14 * d, 10, n), 18, 80)
    income_k = np.clip(rng.lognormal(np.log(55 - 20 * d), 0.35, n), 8, 400)
    debt_ratio = np.clip(rng.normal(0.33 + 0.22 * d, 0.12, n), 0.0, 1.5)
    util = np.clip(rng.normal(0.40 + 0.20 * d, 0.18, n), 0.0, 1.2)
    late = rng.poisson(0.6 + 1.2 * d, n)
    acct_age = np.clip(rng.normal(8 - 3 * d, 4, n), 0.2, 40)
    loan_k = np.clip(rng.lognormal(np.log(15 + 10 * d), 0.5, n), 1, 200)

    # The "true rule" for who defaults. A new pattern appears as drift grows
    # (young borrowers become riskier) = concept drift.
    logit = 1.5 * (-1.4 - 2.3 * d
                   + 3.2 * (debt_ratio - 0.33)
                   + 1.8 * (util - 0.40)
                   + 0.45 * late
                   - 0.035 * (income_k - 55)
                   - 0.04 * (acct_age - 8)
                   + 0.015 * (loan_k - 15)
                   + (d ** 2) * 2.4 * (age < 30)
                   + (d ** 2) * 1.5 * (debt_ratio > 0.5))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"age": age, "income_k": income_k, "debt_ratio": debt_ratio,
                       "credit_utilization": util, "late_payments": late,
                       "account_age_years": acct_age, "loan_amount_k": loan_k})
    if unit_error_rate > 0:
        bad = rng.random(n) < unit_error_rate
        df.loc[bad, ["debt_ratio", "credit_utilization", "late_payments"]] = 0.0  # feed outage -> zero-filled
    if missing_rate > 0:
        mask = rng.random(n) < missing_rate
        df.loc[mask, "income_k"] = np.nan
    return df, y
```

### Explained in plain English

- **`FEATURES`**: the 7 columns (facts) about each customer.
- **`make_data(n, seed, drift, missing_rate, unit_error_rate)`**: creates `n` customers.
  - `seed`: makes the randomness repeatable. Same seed gives the same data.
  - `drift` (0 to 1): how much the world has changed. **0 = same as training, 1 = very different.**
  - `missing_rate`: the share of income values that vanish (a broken data pipeline).
  - `unit_error_rate`: the share of rows where a feed outage zeroes out three columns.
- **The lines with `rng.normal(...)`**: generate each fact. Look at `age`: the average is `42 - 14 * d`. At `d = 0` the average age is 42. At `d = 1` it is 28. So as drift grows, customers get **younger**. The same happens to income (falls), debt ratio (rises), and so on. That is **data drift**.
- **`logit = ...`**: the hidden "true rule" for who defaults. High debt and many late payments raise the chance. High income lowers it. The two last lines (with `d ** 2`) add **a new pattern that only appears as drift grows**: young borrowers become riskier. The model never learned this, so its accuracy falls. That is **concept drift**. Using `d ** 2` means this effect is tiny at first and large later, so accuracy drops *after* the data has already started shifting, like in real life.
- **`y = ...`**: turns the risk score into a 0/1 answer by a weighted coin flip.
- **The `unit_error_rate` block**: for a random share of rows, sets debt ratio, utilisation and late payments to 0, like a broken feed that sends zeros.
- **The `missing_rate` block**: blanks out income for a random share of rows.
- **`return df, y`**: gives back the table of customers and their true answers.

---

## 4. File 2: `train.py` (trains the target model and saves the "fingerprint")

**Job:** (a) train the test-subject model, (b) save a **baseline fingerprint**: a summary of what "normal" looked like at training time.

```python
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
```

### Explained in plain English

- **`entropy(proba)`**: a number for how *unsure* the model is. If it says "50% default, 50% safe", entropy is at its highest (1.0). If it says "99% safe", entropy is near 0.
- **`make_data(20000, seed=1)`**: creates 20,000 customers with **no drift** (the "old world").
- **`train_test_split`**: keeps 75% to train and 25% aside to test. The 25% gives us an honest "normal accuracy" (about 82%).
- **`medians`**: the typical value of each column. Later, if a value is missing, the pipeline fills it with this typical value (a very common real-world practice).
- **`RandomForestClassifier(...)`**: the target model. Think of it as 150 small decision trees voting. You never need to tune this. It is just a test subject.
- **`proba = model.predict_proba(...)`**: for each customer, the model's probability of default (e.g., 0.83 = 83%). **Confidence** = the highest probability (the model's "how sure am I").
- **`ref_acc`**: the model's normal accuracy. Used only by the test script.
- **The `for f in FEATURES` loop builds the fingerprint, one feature at a time:**
  - `edges`: cuts the values into **10 equal groups** (deciles). So in normal times each group holds 10% of customers.
  - `props`: the share of customers in each group (about 0.1 each).
  - `lo` / `hi`: the 0.5th and 99.5th percentile. Values outside this range are "unusually small or large".
  - `missing_rate`: how often the column was empty in normal times.
- **`baseline = {...}`**: the whole fingerprint. It also stores normal average confidence, normal entropy and the normal share of "default" predictions.
- **Saved files:** `model/model.joblib` (the model) and `baseline/baseline.json` (the fingerprint).

**Privacy point (great for judges):** the fingerprint is only counts and averages. **No customer rows are stored.** Open `baseline/baseline.json` to see it.

---

## 5. File 3: `monitor.py` (the heart of the project)

**Job:** take one batch of incoming data plus the model's answers, and produce scores and a warning. Read the code first, then the part-by-part explanation.

```python
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


def check_batch(X, proba, baseline):
    """Score one batch. X = feature DataFrame, proba = model's predicted probabilities."""
    feats = baseline["features"]

    # ---- 1. Distribution drift (PSI per feature) ----
    psi = {}
    for f, b in feats.items():
        col = X[f].dropna().values
        edges = np.array([-np.inf if e is None and i == 0 else (np.inf if e is None else e)
                          for i, e in enumerate(b["edges"])])
        counts, _ = np.histogram(col, bins=edges)
        psi[f] = _psi(b["props"], counts / max(counts.sum(), 1))
    top3 = sorted(psi.values(), reverse=True)[:3]
    drift_score = 100 * min(1.0, np.mean(top3) / 0.35)

    # ---- 2. Data quality ----
    miss_inc = max(X[f].isna().mean() - b["missing_rate"] for f, b in feats.items())
    out_inc = []
    for f, b in feats.items():
        col = X[f].dropna()
        out_inc.append(((col < b["lo"]) | (col > b["hi"])).mean() - 0.01)
    quality_score = 100 * min(1.0, max(miss_inc / 0.15, max(out_inc) / 0.20, 0))

    # ---- 3. Model behaviour (no labels needed) ----
    conf = float(proba.max(axis=1).mean())
    ent = _entropy(proba)
    pos = float((proba.argmax(axis=1)).mean())
    conf_drop = max(0.0, baseline["confidence_mean"] - conf)
    ent_rise = max(0.0, ent - baseline["entropy_mean"])
    pos_shift = abs(pos - baseline["positive_rate"])
    behavior_score = 100 * min(1.0, (min(1, conf_drop / 0.06) + min(1, ent_rise / 0.08)
                                     + min(1, pos_shift / 0.12)) / 2.0)

    weighted = (WEIGHTS["drift"] * drift_score + WEIGHTS["quality"] * quality_score
                + WEIGHTS["behavior"] * behavior_score)
    # A single very strong signal (e.g. broken data feed) must not be diluted by calm ones.
    raw = max(weighted, 0.7 * max(drift_score, quality_score, behavior_score))
    return {"drift_score": drift_score, "quality_score": quality_score,
            "behavior_score": behavior_score, "risk_raw": raw,
            "confidence": conf, "entropy": ent, "positive_rate": pos,
            "psi": psi, "missing_increase": float(miss_inc)}


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
    parts = {"drift": result["drift_score"], "quality": result["quality_score"],
             "behavior": result["behavior_score"]}
    top_feats = [f for f, _ in sorted(result["psi"].items(), key=lambda kv: -kv[1])[:3]]
    main = max(parts, key=parts.get)
    if max(parts.values()) < WARN:
        return "No significant issue detected.", "Keep monitoring.", "none"
    if main == "quality":
        return ("Likely a DATA-QUALITY problem (missing or out-of-range values).",
                "Check the data pipeline first; retraining is probably not needed.", "quality")
    if main == "drift":
        return (f"Likely real-world DRIFT. Most-shifted features: {', '.join(top_feats)}.",
                "Start collecting fresh labelled data and plan a retrain.", "drift")
    return ("The model is behaving differently (confidence/outputs changing).",
            "Route borderline predictions to human review and investigate.", "behavior")
```

### Part A: the settings at the top

```python
WEIGHTS = {"drift": 0.45, "quality": 0.20, "behavior": 0.35}
WARN, HIGH, DANGER = 35, 55, 75
EMA_ALPHA = 0.5
```

- `WEIGHTS`: how much each signal counts in the final risk score (drift counts most).
- `WARN, HIGH, DANGER`: the risk levels. Below 35 = healthy, 35-55 = watch, 55-75 = high, 75+ = danger.
- `EMA_ALPHA`: smoothing, explained in Part F.

### Part B: PSI, the drift measuring stick (`_psi`)

**PSI** (Population Stability Index) says how different today's data is from normal, as one number.

Simple example. In normal times, 10% of customers have an income in the lowest group. This week, 25% do.

```
(new share - old share) x ln(new share / old share)
= (0.25 - 0.10) x ln(2.5) = 0.15 x 0.92 = 0.137
```

The code does this for all 10 groups and adds them up. Rule of thumb: **under 0.1 = stable, 0.1 to 0.25 = moderate shift, over 0.25 = big shift.** The small `eps` just prevents a mathematical error when a group is empty.

### Part C: `check_batch`, step 1: drift score

For every feature the code counts how many values land in each of the 10 normal groups (`np.histogram`), compares to the normal 10% each (PSI), and keeps one PSI number per feature in the dictionary `psi`.

```python
top3 = sorted(psi.values(), reverse=True)[:3]
drift_score = 100 * min(1.0, np.mean(top3) / 0.35)
```

It takes the **three most-shifted features**, averages their PSI, and scales it to 0-100. An average PSI of 0.35 or more means 100. Why top 3 and not all 7? Because real drift often hits only a few features, and averaging all of them would water the signal down.

### Part D: step 2: data-quality score

Two checks:
1. **Missing values:** how much more empty is each column than normal? (`miss_inc` takes the worst column.)
2. **Out-of-range values:** what share of values falls outside the normal range (`lo` to `hi`)? In normal times about 1% do (that is how `lo` and `hi` were set), so the code subtracts 0.01.

```python
quality_score = 100 * min(1.0, max(miss_inc / 0.15, max(out_inc) / 0.20, 0))
```

So **15% extra missing values**, or **20% extra out-of-range values**, already gives the full 100. A broken data feed is serious, so the scale is strict.

### Part E: step 3: model behaviour score (no answers needed)

Three things are measured from the model's own outputs:

| Signal | Normal | Danger sign | Scaled by |
|---|---|---|---|
| Average confidence | e.g., 0.80 | falls | a drop of 0.06 = full marks |
| Average entropy (unsureness) | e.g., 0.65 | rises | a rise of 0.08 = full marks |
| Share of "default" predictions | e.g., 20% | changes either way | a shift of 12 points = full marks |

```python
behavior_score = 100 * min(1.0, (a + b + c) / 2.0)
```

where a, b, c are the three signals each capped at 1. Dividing by 2 means that **two signals fully triggered already give 100**, so it reacts quickly.

### Part F: combining into one risk number

```python
weighted = 0.45*drift + 0.20*quality + 0.35*behavior
raw = max(weighted, 0.7 * max(drift_score, quality_score, behavior_score))
```

First a weighted average. Then a safety rule: **one very strong signal must not be hidden by two calm ones.** If the data feed is badly broken (quality = 100) but drift and behaviour are calm, a plain average would say "only 20". The `max(...)` line makes sure the risk is at least 70% of the strongest signal (here 70).

### Part G: `RiskTracker`, smoothing and forecasting

```python
smooth = 0.5 * new_value + 0.5 * previous_value
```

This is **smoothing** (called EMA). One random jumpy batch should not trigger an alarm, so each score is half new, half previous.

**The forecast** (`forecast`):
1. Look at the last 6 smoothed risk values.
2. Fit a straight line through them (`np.polyfit`). The **slope** says how fast risk is rising per batch.
3. If risk is already 75 or more, the answer is 0 ("already in danger").
4. If the slope is under 0.5 per batch, it is not really rising, so no forecast is made.
5. Otherwise: **batches left = (75 - current risk) / slope.**

Example: risk is 45 now and rising 6 points per batch. (75 - 45) / 6 = **5 batches** until danger. This is the "predictive" part of the project.

### Part H: `level` and `diagnose`

- `level(risk)`: turns the number into a word: HEALTHY, WATCH, HIGH or DANGER.
- `diagnose(result)`: finds which of the three scores is highest and writes a plain-English cause and action:

| Highest score | Message | Suggested action |
|---|---|---|
| Quality | Likely a data-quality problem | Fix the data pipeline first; retraining probably not needed |
| Drift | Likely real-world drift (names the 3 most-shifted features) | Collect fresh data and plan a retrain |
| Behaviour | The model is behaving differently | Send borderline cases to a human and investigate |

If all three are below 35 it says "No significant issue". It also returns a short tag (`quality`, `drift`, `behavior`, `none`) used by the dashboard.

> The diagnosis is a **likely cause**, not a certainty. Say it that way in your pitch.

---

## 6. File 4: `simulate.py` (plays 40 days of time and grades the monitor)

**Job:** pretend time is passing, feed batches to the model and monitor, and then check whether the warning came before accuracy actually dropped.

```python
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


if __name__ == "__main__":
    main()
```

### Explained in plain English

**The settings**
- 40 batches of 500 customers each.
- Drift begins at batch 10 and grows until batch 34. So batches 0-9 are normal.
- "Failure" is defined as: accuracy (averaged over 3 batches) falls **5 points below normal**.

**`schedule(scenario, i)`**: says how much the world has changed at batch `i`. `ramp` goes from 0 to 1 over batches 10-34. There are three scenarios:

| Scenario | What happens | Why it exists |
|---|---|---|
| `control` | Nothing changes | Proves the monitor does **not** cry wolf |
| `gradual_drift` | The world slowly changes (younger, poorer, more debt customers) | The main test: does it warn early? |
| `quality_break` | A feed outage zero-fills columns, and some income values go missing | Shows the diagnosis: "data-quality problem" |

**`run(...)`**: the main loop, once per batch:
1. Make a batch of customers (`make_data`) with the right amount of drift.
2. Fill missing values with the normal typical values (`fillna`), exactly as a real pipeline would, then let the model predict.
3. **`check_batch(X, proba, baseline)`**: the monitor scores the batch. *Look at what is passed in: the data and the model's probabilities. No labels.*
4. `tracker.update(...)`: smooth the risk and get the forecast.
5. `diagnose(...)`: write the cause and action.
6. **Only now** is the true accuracy computed (`acc = ...`) using the true answers `y`. This is for grading only.
7. Everything is saved as one row. After 40 rows, the 3-batch rolling accuracy is added.

**`evaluate(...)`** finds the key numbers:
- **Failure batch:** the first batch where rolling accuracy is more than 5 points under normal.
- **Alert batch:** the first batch where risk was at least 35 for **2 batches in a row** (two in a row avoids false alarms).
- **Lead time = failure batch - alert batch.** If the alert came at batch 14 and failure at batch 20, you warned **6 batches early**.

**`main()`**: runs every scenario with **10 different random seeds** (10 repeats), so the result is not a lucky one-off. It counts:
- For `control`: the **false-alarm rate** (should be 0%).
- For `gradual_drift`: in how many runs the warning came before the failure, and the median lead time.
- For `quality_break`: how many runs raised an alert. Accuracy barely drops in this scenario, so **no lead-time claim is made for it**. Be honest about that.

It saves `results/results.csv` (the detailed first run of every scenario, used by the dashboard), `results/backtest_runs.csv` and `results/summary.json`.

### The results I got when I tested it

| Test | Result |
|---|---|
| Gradual drift | Warned before failure in **10 of 10** runs, median lead time **5.5 batches** (minimum 4) |
| Control (no drift) | **0%** false alarms |
| Broken feed | Alert raised in 10 of 10 runs and diagnosed as data quality; accuracy barely changed, so no lead-time claim |

Your numbers should match, because the seeds are fixed.

---

## 7. File 5: `app.py` (the dashboard)

**Job:** show the results visually so judges can *see* the warning arrive before the accuracy drop.

```python
"""Step 3: the dashboard.  Run with:  streamlit run app.py"""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitor import WARN, HIGH, DANGER

st.set_page_config(page_title="AERO", page_icon="🛡️", layout="wide")
st.title("🛡️ AERO: predictive model monitoring")
st.caption("Warns BEFORE accuracy falls, without needing the true labels. "
           "Only statistical summaries are kept; raw records are never stored.")

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
```

### Explained in plain English

- **Streamlit** turns a Python script into a web page. Each `st.something(...)` line adds one element to the page.
- **The `try` block** loads the results. If the files are missing it shows the message "Run train.py and simulate.py first".
- **The sidebar** has three controls: the scenario picker, the **time slider** (which batch you are looking at), and a checkbox to show the hidden true accuracy.
- **Top-left: the gauge** shows the risk score at the chosen time, coloured by level.
- **Top-right: the status panel** shows the level, the forecast ("risk reaches danger in about N batches"), the diagnosis, the suggested action and the three sub-scores.
- **"Risk over time" chart:** the thick blue line is the risk score; dotted lines are the three sub-scores; the **black line is the true accuracy**, which the monitor never saw. **This is your key chart.** The blue line rises *before* the black line falls. That gap is your lead time.
- **"Which features shifted most":** bars of each feature's PSI, coloured green/yellow/red. It tells the team *where* the problem is.
- **"Backtest proof" at the bottom:** the headline numbers (10/10 warned, median lead time, 0% false alarms) from `summary.json`.

### How to demo it (60 seconds)
1. Pick **Gradual real-world drift** and drag the slider to batch 5: green and healthy.
2. Drag to about batch 12-14: the gauge turns yellow/orange and the forecast appears, **while the black accuracy line is still high**.
3. Drag to about batch 20-25: the black line has now fallen. Say: "We warned several batches before this."
4. Switch to **Broken data feed**: the diagnosis changes to a data-quality problem.
5. Switch to **Control**: it stays green, so there are no false alarms.

---

## 8. Settings you can safely change (and what happens)

| Setting | File | Effect |
|---|---|---|
| `WARN` (35) | `monitor.py` | Lower = earlier warnings but more false alarms |
| `WEIGHTS` | `monitor.py` | Which signal counts most in the risk score |
| `0.35`, `0.15`, `0.20` divisors | `monitor.py` | How sensitive each score is |
| `ACC_DROP` (0.05) | `simulate.py` | How big a drop counts as "failure" |
| `DRIFT_START`, `DRIFT_END` | `simulate.py` | When and how fast the world changes |
| `n_seeds` (10) | `simulate.py` | More repeats = more convincing statistics |

After changing anything, rerun `python simulate.py` and refresh the dashboard. **Do not tune until the numbers look good on one scenario and then claim it works in general.** Be honest that thresholds were hand-tuned.

---

## 9. Common problems and fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named ...` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: baseline.json` or model file | Run `python train.py` first |
| Dashboard says "Run train.py and simulate.py first" | Run `python simulate.py` |
| `streamlit` is not recognised | Use `python -m streamlit run app.py` |
| Error mentioning `width` in `st.plotly_chart` | Your Streamlit is old: run `pip install -U streamlit` |
| Browser doesn't open | Open the address shown in the terminal, usually `http://localhost:8501` |
| Numbers differ from mine | You changed a setting or seed; this is expected |

When stuck, copy the **whole error message** into Antigravity and say: *"Explain this error in simple words and fix it."*

---

## 10. Honest limitations (say these before the judges find them)

1. **The data is synthetic.** It lets us control and measure the drift. A real, time-split dataset would make the claim stronger.
2. **Lead time is relative to our definition** of failure (accuracy 5 points below normal, over 3 batches).
3. **Thresholds and weights were tuned by hand** on this simulation.
4. **Label-free monitoring cannot catch every failure.** If the real rules change but inputs and confidence look the same, we may miss it.
5. **The data-outage scenario** triggers an alert and a diagnosis, but accuracy barely moved, so we do not claim lead time there.
6. **Privacy:** the monitor keeps only statistical summaries, but it still briefly processes the data and must be secured like any pipeline component.

## 11. Likely judge questions, short answers

**"How is this different from Evidently or NannyML?"** They detect or estimate after a shift. We combine three signal types into one risk score, forecast time-to-danger, and validate the lead time with repeated backtests.

**"How do you know the model is failing without labels?"** We don't. We estimate risk from signals that move before accuracy does, and the backtest shows how reliable that is.

**"Is the data real?"** No, it is synthetic, so the drift is controlled and the lead time measurable. The framework works on any tabular model.

**"What about privacy?"** The monitor runs next to the model, stores only histograms and averages, and never keeps raw records.

**"What can the team do after an alert?"** Fix the data pipeline, route risky predictions to human review, start collecting data, retrain, or roll back. The diagnosis suggests which.

---

## 12. Glossary

- **Model:** a program that learned patterns from data.
- **Feature:** one input column (age, income...).
- **Label:** the true answer.
- **Batch:** a chunk of data processed together.
- **Drift:** the data changing over time.
- **PSI:** a number measuring how much a distribution has shifted.
- **Confidence:** how sure the model is about its answer.
- **Entropy:** a measure of how unsure the model is.
- **Baseline / fingerprint:** a summary of what normal looked like at training time.
- **Risk score:** the combined 0-100 warning number.
- **Lead time:** how many batches early the warning came before accuracy fell.
- **Backtest:** replaying a simulated history to check the system works.
- **EMA (smoothing):** blending new and old values so one noisy batch does not cause an alarm.
- **False alarm:** a warning when nothing was actually wrong.
