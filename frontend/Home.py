# streamlit_scoring.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

st.set_page_config(page_title="MC Scoring & Ranking", layout="wide")

# -----------------------
# Backend endpoint
# -----------------------
BACKEND_URL = st.sidebar.text_input("Backend simulate URL", value="http://127.0.0.1:8000/simulate")

# -----------------------
# Vital names (model output order)
# -----------------------
VITALS = [
    "Respiratory Rate","Heart Rate","SpO2","PaO2","Systolic BP",
    "Diastolic BP","MAP","PIP","pH","PaCO2","Hematocrit","Hemoglobin"
]

# -----------------------
# Scoring configuration (defaults taken from your provided config)
# Keys use model-style names: PaCO2, Hematocrit
# -----------------------
VITAL_BOUNDS = {
    "Heart Rate": (60, 100),
    "MAP": (70, 100),
    "Systolic BP": (90, 150),
    "Diastolic BP": (50, 90),
    "SpO2": (0.90, 1.00),
    "PaO2": (80, 400),
    "Respiratory Rate": (12, 24),
    "pH": (7.40, 7.48),
    "PaCO2": (35, 42),
    "Hemoglobin": (10, 16),
    "Hematocrit": (0.30, 0.48),
    "PIP": (15, 25),
}

VITAL_WEIGHTS = {
    "SpO2": 0.10,
    "PaO2": 0.15,
    "pH": 0.06,
    "PaCO2": 0.1,
    "MAP": 0.05,
    "Heart Rate": 0.15,
    "Respiratory Rate": 0.2,
    "PIP": 0.1,
    "Systolic BP": 0.06,
    "Diastolic BP": 0.04,
    "Hemoglobin": 0.05,
    "Hematocrit": 0.05
}

PENALTIES = {
    "SpO2":      {"alpha": 8.0,  "beta": 2.0},
    "PaO2":      {"alpha": 7.0,  "beta": 3.0},
    "pH":        {"alpha": 6.0, "beta": 4.0},
    "PaCO2":     {"alpha": 6.0,  "beta": 6.0},
    "MAP":       {"alpha": 6.0,  "beta": 1.5},
    "Heart Rate":{"alpha": 2.0,  "beta": 7},
    "Respiratory Rate": {"alpha": 2.0, "beta": 7},
    "PIP":       {"alpha": 1.5,  "beta": 6},
    "Systolic BP":{"alpha": 7, "beta": 1.5},
    "Diastolic BP":{"alpha": 6.0,"beta": 1.5},
    "Hemoglobin":{"alpha": 3, "beta": 1.0},
    "Hematocrit":{"alpha": 3, "beta": 1.0}
}

# Strategy columns expected in samples
STRATEGY_COLS = [
    "fio2","peep","inspiratory_pressure","respiration_rate",
    "flow","tidal_volume","Mode_ACP","Mode_ACV","Mode_CMP","Mode_CMV"
]

# -----------------------
# Scoring utilities (exact logic adapted from your provided code)
# -----------------------
def compliance_q(value, vmin, vmax, alpha=5.0, beta=2.0, eps=1e-12):
    import math
    if pd.isna(value):
        return 0.0
    if (value >= vmin) and (value <= vmax):
        return 1.0
    if value < vmin:
        rel = (vmin - value) / max(abs(vmin), eps)
        q = float(np.exp(- alpha * rel))
        return q
    rel = (value - vmax) / max(abs(vmax), eps)
    q = float(np.exp(- beta * rel))
    return q

def compute_physiological_score(df,
                                bounds=VITAL_BOUNDS,
                                weights=VITAL_WEIGHTS,
                                penalties=PENALTIES,
                                score_col_prefix="q_"):
    df = df.copy()
    q_cols = []
    # normalize weights if needed
    w = weights.copy()
    total_w = sum(w.values())
    if abs(total_w - 1.0) > 1e-8 and total_w > 0:
        for k in w:
            w[k] = w[k] / total_w

    for vital, (vmin, vmax) in bounds.items():
        col_name = vital
        score_name = score_col_prefix + vital.replace(" ", "__")
        if col_name not in df.columns:
            continue
        alpha = penalties.get(vital, {}).get("alpha", 5.0)
        beta  = penalties.get(vital, {}).get("beta", 2.0)
        df[score_name] = df[col_name].apply(lambda x: compliance_q(x, vmin, vmax, alpha=alpha, beta=beta))
        q_cols.append((vital, score_name))

    weighted_q = np.zeros(len(df), dtype=float)
    used_weights_sum = 0.0
    for vital, score_name in q_cols:
        wt = w.get(vital, 0.0)
        weighted_q += wt * df[score_name].fillna(0.0).values
        used_weights_sum += wt

    if used_weights_sum > 0:
        df["phys_score"] = weighted_q / used_weights_sum
    else:
        df["phys_score"] = 0.0

    return df

def make_strategy_id(df_row, cols=STRATEGY_COLS):
    parts = []
    for c in cols:
        if c in df_row:
            parts.append(f"{c}={df_row[c]}")
    return "|".join(parts)

def topk_strategies_per_severity(df, k=3,
                                severity_cols=("left_severity","right_severity","brain_injury_severity"),
                                strategy_cols=STRATEGY_COLS,
                                score_col="phys_score"):
    required = list(severity_cols) + strategy_cols + [score_col]
    present = [c for c in required if c in df.columns]
    group_cols = [c for c in severity_cols if c in df.columns] + [c for c in strategy_cols if c in df.columns]
    if not group_cols:
        raise ValueError("No grouping columns present. Check severity_cols and strategy_cols vs dataframe columns.")
    g = df.groupby(group_cols)[score_col].agg(["mean","std","count"]).reset_index().rename(columns={"mean":"mean_score","std":"std_score","count":"n"})
    severity_keys = [c for c in severity_cols if c in df.columns]
    sort_cols = severity_keys + ["mean_score"]
    g = g.sort_values(by=sort_cols, ascending= [True]*len(severity_keys) + [False]).reset_index(drop=True)
    topk = g.groupby(severity_keys).head(k).reset_index(drop=True)
    return topk

# -----------------------
# Helper: convert one-hot to Mode
# -----------------------
def onehot_modes_to_col(samples_df):
    mode_cols = [c for c in ["Mode_ACP","Mode_ACV","Mode_CMP","Mode_CMV"] if c in samples_df.columns]
    if not mode_cols:
        return samples_df
    def pick_mode(row):
        for c in mode_cols:
            if int(row.get(c, 0)) == 1:
                return c.replace("Mode_", "")
        # fallback: if none are 1, return 'Unknown'
        return "Unknown"
    samples_df = samples_df.copy()
    samples_df["Mode"] = samples_df.apply(pick_mode, axis=1)
    return samples_df

# -----------------------
# UI: Controls
# -----------------------
st.title("Monte-Carlo → Scoring → Top-10 Strategies")
st.markdown("Run Monte-Carlo, score model predictions, and rank strategies. Uses clinical compliance scoring.")

st.sidebar.header("Simulation Inputs")
left_sev = st.sidebar.number_input("Left Lung Severity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
right_sev = st.sidebar.number_input("Right Lung Severity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
brain_sev = st.sidebar.number_input("Brain Injury Severity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Ventilator Feature Bounds (provide realistic ranges)")

VENTILATOR_UNITS = {
    "fio2": "%",
    "peep": "cmH₂O",
    "inspiratory_pressure": "cmH₂O",
    "respiration_rate": "breaths/min",
    "flow": "L/min",
    "tidal_volume": "mL"
}


# Default example ranges — user can edit
default_bounds = {
    "fio2": (21.0, 100.0),
    "peep": (5.0, 12.0),
    "inspiratory_pressure": (10.0, 25.0),
    "respiration_rate": (10.0, 20.0),
    "flow": (10.0, 20.0),
    "tidal_volume": (300.0, 500.0)
}
# ------------------------------------
# Ventilator feature range sliders
# ------------------------------------
bounds = {}

for key, (low, high) in default_bounds.items():
    unit = VENTILATOR_UNITS.get(key, "")
    
    bounds[key] = list(
        st.sidebar.slider(
            f"{key.replace('_',' ').title()} Range ({unit})",
            min_value=float(low),
            max_value=float(high),
            value=(float(low), float(high))
        )
    )

# Convert FiO₂ from % → fraction (model expects 0–1)
if "fio2" in bounds:
    bounds["fio2"] = (bounds["fio2"][0] / 100, bounds["fio2"][1] / 100)


st.sidebar.markdown("---")
N = st.sidebar.number_input("Number of Monte Carlo Samples", min_value=100, max_value=20000, value=2000, step=100)

show_plots = st.sidebar.checkbox("Show Score Distribution Plot", value=True)
editable_scoring = st.sidebar.checkbox("Allow editing weights & penalties", value=False)

# -----------------------
# Optional scoring editor
# -----------------------
if editable_scoring:
    with st.expander("Edit weights (will be re-normalized)"):
        for k in list(VITAL_WEIGHTS.keys()):
            VITAL_WEIGHTS[k] = float(st.number_input(f"Weight: {k}", value=float(VITAL_WEIGHTS[k]), key="w_"+k))
    with st.expander("Edit penalties (alpha=low-side, beta=high-side)"):
        for k in list(PENALTIES.keys()):
            a = float(st.number_input(f"alpha {k}", value=float(PENALTIES[k]["alpha"]), key="a_"+k))
            b = float(st.number_input(f"beta {k}", value=float(PENALTIES[k]["beta"]), key="b_"+k))
            PENALTIES[k]["alpha"] = a
            PENALTIES[k]["beta"] = b

# -----------------------
# Run simulation button
# -----------------------
if st.button("🚀 Run Monte-Carlo & Score (Top 10)"):
    payload = {
        "left_sev": float(left_sev),
        "right_sev": float(right_sev),
        "brain_sev": float(brain_sev),
        "N": int(N),
        "bounds": bounds
    }

    st.info("Posting to backend simulate endpoint...")
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=300)
    except Exception as e:
        st.error("Could not reach backend.")
        st.exception(e)
        st.stop()

    if resp.status_code != 200:
        st.error(f"Backend returned status {resp.status_code}: {resp.text}")
        st.stop()

    resp_json = resp.json()

    # Backend compatibility:
    # prefer 'samples' and 'predictions' keys, but support older 'predictions' only format.
    samples_df = None
    if "samples" in resp_json:
        samples_df = pd.DataFrame(resp_json["samples"])
    elif "samples_count" in resp_json and "predictions" in resp_json:
        # no samples returned — we only have predictions; samples_count present
        samples_df = None
    # predictions: model outputs (list of lists)
    raw_preds = resp_json.get("predictions", [])
    preds_df = pd.DataFrame(raw_preds, columns=VITALS)

    # if samples returned, convert mode one-hot to Mode column and combine
    if samples_df is not None:
        samples_df = onehot_modes_to_col(samples_df)

        # combine side-by-side: predictions first, then sample columns
        combined = pd.concat([samples_df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
    else:
        # no samples available from backend — create combined only with preds (cannot show parameters)
        combined = preds_df.copy()

    # Score using compute_physiological_score (adds q_... columns + phys_score)
    scored = compute_physiological_score(combined)

    # If Mode not present but mode onehots present in columns of scored, still add Mode
    if "Mode" not in scored.columns and any(c in scored.columns for c in ["Mode_ACP","Mode_ACV","Mode_CMP","Mode_CMV"]):
        scored = onehot_modes_to_col(scored)

    # Sort and show top 10 by phys_score (descending)
    if "phys_score" not in scored.columns:
        st.error("Scoring failed — 'phys_score' not computed.")
        st.stop()

    topk = scored.sort_values("phys_score", ascending=False).head(10).reset_index(drop=True)

    st.success("Scoring completed — showing Top 10 strategies (by phys_score).")

    # Display top 10 table with key columns: strategy params, Mode, vitals, phys_score
    show_cols = []
    # prefer showing strategy columns if present
    for c in STRATEGY_COLS:
        if c in topk.columns:
            show_cols.append(c)
    # show Mode if present
    if "Mode" in topk.columns:
        show_cols.append("Mode")
    # then vitals
    for v in VITALS:
        if v in topk.columns:
            show_cols.append(v)
    # final score
    show_cols.append("phys_score")

    st.subheader("🏆 Top 10 Ranked Monte-Carlo Samples")
    st.dataframe(topk[show_cols], use_container_width=True)

    # Download scored CSV
    csv_bytes = topk.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top 10 CSV", csv_bytes, "top10_scored.csv", mime="text/csv")

    # Full scored dataset download
    all_csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Scored Dataset", all_csv, "scored_full.csv", mime="text/csv")

    # Score distribution plot
    if show_plots:
        st.subheader("Score distribution (all Monte-Carlo rows)")
        fig = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,3))
            ax.hist(scored["phys_score"].dropna(), bins=40)
            ax.set_xlabel("phys_score")
            ax.set_ylabel("count")
            st.pyplot(fig)
        except Exception:
            st.line_chart(scored["phys_score"])

    # Per-mode top-3 tables (if Mode present)
    if "Mode" in scored.columns:
        st.subheader("Top 3 per Mode")
        modes = scored["Mode"].unique()
        for m in modes:
            st.markdown(f"**Mode: {m}**")
            t = scored[scored["Mode"] == m].sort_values("phys_score", ascending=False).head(3)
            st.dataframe(t[show_cols], use_container_width=True)

    # Also show per-vital q_ columns for the top10 in an expander
    st.subheader("Per-vital compliance (top 10)")
    q_cols = [c for c in topk.columns if c.startswith("q_")]
    if q_cols:
        st.dataframe(topk[q_cols], use_container_width=True)
    else:
        st.info("No per-vital q_ columns found — maybe some vitals are missing from the model output.")
