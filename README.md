# AERO: predictive model monitoring (hackathon prototype)

Warns BEFORE a deployed model's accuracy drops, using only signals available without true labels:
input drift (PSI), data quality (missing / out-of-range), and model behaviour (confidence, entropy, output mix).
These are combined into a 0-100 risk score, trend-forecast to estimate time-to-danger, and backtested for lead time.

## Run
    pip install -r requirements.txt
    python train.py        # trains the target model + saves the baseline fingerprint
    python simulate.py     # simulates time passing + runs the backtest (10 seeds x 3 scenarios)
    streamlit run app.py   # dashboard

## Files
- data_gen.py : synthetic credit-risk data with controllable drift / data-feed faults
- train.py    : target model (RandomForest) + baseline summaries (no raw rows kept)
- monitor.py  : the monitoring engine. NEVER receives labels.
- simulate.py : replays batches; labels are used here ONLY to measure true accuracy for the backtest
- app.py      : Streamlit dashboard

## Honest limitations
- Data is synthetic; swap in a real time-split dataset for stronger claims.
- Lead time is measured against accuracy falling 5 points below reference (3-batch rolling).
- Thresholds/weights were tuned by hand on this simulation, so test on new scenarios before generalising.
- Label-free monitoring cannot catch concept drift that leaves inputs and confidence unchanged.
- The data-outage scenario triggers alerts but barely changes accuracy here, so no lead-time claim for it.
