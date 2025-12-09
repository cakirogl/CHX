import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Helper Functions
# ==========================================
st.set_page_config(page_title="CHX Release Predictor", layout="wide")

# URLs for data (used for calibration/conformal prediction)
URL_CUMULATIVE = "https://raw.githubusercontent.com/cakirogl/CHX/main/CHX_elution_cumulative_all_doses.csv"
URL_RATE = "https://raw.githubusercontent.com/cakirogl/CHX/main/chx_all_doses_rate.csv"

# Model Paths (Assumed local paths based on previous steps)
PATH_MODEL_CUMULATIVE = "./xgb_elution_outputs/xgb_elution_optimized.pkl"
PATH_MODEL_RATE = "./xgb_elution_outputs/rate_xgb_optimized.pkl"

def make_features(day: np.ndarray, dose: np.ndarray) -> np.ndarray:
    """
    Feature engineering pipeline. Must match training exactly.
    """
    t = day.astype(float)
    d = dose.astype(float)

    sqrt_t = np.sqrt(t)
    log1p_t = np.log1p(t)
    t_pow03 = np.power(t, 0.3)
    t2 = np.power(t, 2.0)

    td = t * d
    sqrt_t_d = sqrt_t * d
    log1p_t_d = log1p_t * d
    t2_d = t2 * d
    d2 = np.power(d, 2.0)

    X = np.column_stack([
        t, d, sqrt_t, log1p_t, t_pow03, t2,
        td, sqrt_t_d, log1p_t_d, t2_d, d2
    ])
    return X

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    # Handle if pickle is dict or direct model
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload

@st.cache_data
def load_data(url, data_type="rate"):
    """
    Loads and cleans data for calibration.
    """
    df = pd.read_csv(url)
    lower_map = {c.lower(): c for c in df.columns}
    
    # Identify Day
    day_col = next((lower_map[k] for k in ["day", "days_of_testing"] if k in lower_map), None)
    # Identify Dose
    dose_col = lower_map.get("dose")
    
    # Identify Target
    if data_type == "rate":
        target_col = next((lower_map[k] for k in ["releaserate", "release_rate", "rate"] if k in lower_map), None)
    else:
        target_col = next((lower_map[k] for k in ["elution", "cumulative", "elution_cumulative"] if k in lower_map), None)

    if not (day_col and dose_col and target_col):
        return pd.DataFrame() # Return empty if cols missing

    df = df[[day_col, dose_col, target_col]].copy()
    df.columns = ["Day", "Dose", "Target"]
    df = df.dropna().sort_values(["Dose", "Day"]).reset_index(drop=True)
    return df

# ==========================================
# 2. Sidebar Inputs
# ==========================================
st.sidebar.header("Prediction Settings")

# A. Select Prediction Type
pred_type = st.sidebar.selectbox(
    "Select Prediction Target",
    ("Daily Release Rate", "Cumulative Release")
)

# B. Select Dose
dose_input = st.sidebar.selectbox(
    "Select Dose (%)",
    (1, 2, 5, 10)
)

# C. Select Day (Input for Point Prediction)
day_input = st.sidebar.number_input(
    "Select Day",
    min_value=0.0, max_value=40.0, value=5.0, step=0.5
)

# D. Select Quantile (Confidence Level)
confidence_level = st.sidebar.selectbox(
    "Prediction Interval (Confidence)",
    (0.80, 0.90, 0.95, 0.99),
    index=2 # Default to 0.95
)

# ==========================================
# 3. Main Logic
# ==========================================
st.title("CHX Release Prediction App")
st.markdown(f"### Target: **{pred_type}** | Dose: **{dose_input}%**")

# Determine paths and URLs based on selection
if pred_type == "Cumulative Release":
    model_path = PATH_MODEL_CUMULATIVE
    data_url = URL_CUMULATIVE
    data_mode = "cumulative"
    unit = "μmol/m²"
else:
    model_path = PATH_MODEL_RATE
    data_url = URL_RATE
    data_mode = "rate"
    unit = "μmol/m²/day"

# Load Resources
model = load_model(model_path)
df_calib = load_data(data_url, data_mode)

if model is None:
    st.error(f"Model file not found at `{model_path}`. Please ensure the model is trained and saved.")
    st.stop()

if df_calib.empty:
    st.error("Could not load calibration data. Check internet connection or URLs.")
    st.stop()

# ------------------------------------------
# 4. Conformal Prediction (Calibration)
# ------------------------------------------
# We calculate residuals on the *entire* available dataset to determine the interval width
# (In a strict setup, this should be a held-out calibration set, but here we use available data)
X_calib = make_features(df_calib["Day"].values, df_calib["Dose"].values)
y_calib_true = df_calib["Target"].values
y_calib_pred = model.predict(X_calib)

# Absolute residuals (Non-conformity scores)
residuals = np.abs(y_calib_true - y_calib_pred)

# Calculate Quantile (q)
# For 95% confidence, we want the q value where 95% of errors fall below it.
q_val = np.quantile(residuals, confidence_level)

# ------------------------------------------
# 5. Make Prediction for User Input
# ------------------------------------------
X_new = make_features(np.array([day_input]), np.array([dose_input]))
pred_point = model.predict(X_new)[0]

# Apply Interval
lower_bound = max(0, pred_point - q_val) # Physics constraint: cannot be negative
upper_bound = pred_point + q_val

# ------------------------------------------
# 6. Display Results
# ------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Predicted Value", value=f"{pred_point:.4f} {unit}")

with col2:
    st.metric(label="Lower Bound", value=f"{lower_bound:.4f} {unit}", delta=f"-{q_val:.4f}")

with col3:
    st.metric(label="Upper Bound", value=f"{upper_bound:.4f} {unit}", delta=f"+{q_val:.4f}")

st.info(f"Prediction Interval based on **{confidence_level*100:.0f}%** quantile of model residuals (±{q_val:.4f}).")

# ------------------------------------------
# 7. Visualization
# ------------------------------------------
st.markdown("---")
st.subheader("Prediction Curve & Interval")

# Generate curve for plotting
t_plot = np.linspace(0, 35, 200)
d_plot = np.full_like(t_plot, dose_input)
X_plot = make_features(t_plot, d_plot)
y_plot = model.predict(X_plot)

y_lower = np.maximum(0, y_plot - q_val)
y_upper = y_plot + q_val

# Filter actual data for this dose to overlay
df_dose = df_calib[df_calib["Dose"] == dose_input]

fig, ax = plt.subplots(figsize=(10, 5))

# Plot Interval Band
ax.fill_between(t_plot, y_lower, y_upper, color='purple', alpha=0.2, label=f'{int(confidence_level*100)}% Prediction Interval')

# Plot Mean Curve
ax.plot(t_plot, y_plot, color='purple', linewidth=2, label='Model Prediction')

# Plot Actual Data Points
if not df_dose.empty:
    ax.scatter(df_dose["Day"], df_dose["Target"], color='black', s=30, label='Actual Data', zorder=3)

# Plot User Selected Point
ax.scatter([day_input], [pred_point], color='red', s=100, marker='*', label='Your Selection', zorder=4)
ax.errorbar([day_input], [pred_point], yerr=[[pred_point-lower_bound], [upper_bound-pred_point]], fmt='none', ecolor='red', capsize=5, zorder=4)

ax.set_xlabel("Days")
ax.set_ylabel(f"{pred_type} ({unit})")
ax.set_title(f"Dose {dose_input}%: {pred_type} over Time")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
