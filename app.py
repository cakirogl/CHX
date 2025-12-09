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

# URLs for data (used for plotting actual data points)
URL_CUMULATIVE = "https://raw.githubusercontent.com/cakirogl/CHX/main/CHX_elution_cumulative_all_doses.csv"
URL_RATE = "https://raw.githubusercontent.com/cakirogl/CHX/main/chx_all_doses_rate.csv"

# Model Paths
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
    try:
        import dill
        # Load the file
        payload = joblib.load(path)
        
        # Check if it's a dictionary containing the model
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        
        # Otherwise, assume it's the model itself
        return payload
        
    except ImportError:
        # Fallback if dill is missing
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        return payload
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data(url, data_type="rate"):
    """
    Loads and cleans data for plotting actual points.
    """
    try:
        df = pd.read_csv(url)
    except:
        return pd.DataFrame()

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
        return pd.DataFrame()

    df = df[[day_col, dose_col, target_col]].copy()
    df.columns = ["Day", "Dose", "Target"]
    df = df.dropna().sort_values(["Dose", "Day"]).reset_index(drop=True)
    return df

def get_rate_quantile(dose, confidence_level):
    """
    Loads the specific text file for the selected dose and calculates the quantile.
    Returns None if file is missing.
    """
    filename = f"all_scores_xgb_{dose}.txt"
    
    if os.path.exists(filename):
        try:
            scores = np.loadtxt(filename)
            # Calculate quantile (e.g., 95% of errors are below this value)
            q_val = np.quantile(scores, confidence_level)
            return q_val, f"Loaded from {filename}"
        except Exception:
            return None, "Error reading score file"
    else:
        return None, f"Score file {filename} not found"

# ==========================================
# 2. Sidebar Inputs
# ==========================================
st.sidebar.header("Prediction Settings")

pred_type = st.sidebar.selectbox(
    "Select Prediction Target",
    ("Daily Release Rate", "Cumulative Release")
)

dose_input = st.sidebar.selectbox(
    "Select Dose (%)",
    (1, 2, 5, 10)
)

# --- Synchronized Day Input (Slider + Number Box) ---
if 'day_val' not in st.session_state:
    st.session_state.day_val = 5.0

def update_slider():
    st.session_state.day_val = st.session_state.day_slider

def update_num_input():
    st.session_state.day_val = st.session_state.day_num

st.sidebar.markdown("### Select Day")
# 1. Slider
st.sidebar.slider(
    "Use Slider",
    min_value=0.0, max_value=663.0, step=1.0,
    key='day_slider', value=st.session_state.day_val,
    on_change=update_slider
)
# 2. Manual Entry
st.sidebar.number_input(
    "Enter Manually",
    min_value=0.0, max_value=663.0, step=1.0,
    key='day_num', value=st.session_state.day_val,
    on_change=update_num_input
)

# Use the synchronized value
day_input = st.session_state.day_val
# ----------------------------------------------------

# Only show confidence level selection if we are in Rate mode
if pred_type == "Daily Release Rate":
    confidence_level = st.sidebar.selectbox(
        "Prediction Interval (Confidence)",
        (0.80, 0.90, 0.95, 0.99),
        index=2
    )
else:
    confidence_level = None

# ==========================================
# 3. Main Logic
# ==========================================
st.title("CHX Release Predictor")
st.markdown(f"### Target: **{pred_type}** | Dose: **{dose_input}%**")

# Setup paths
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
    st.error(f"Model file not found at `{model_path}`.")
    st.stop()

# ------------------------------------------
# 4. Calculate Prediction & Intervals
# ------------------------------------------
# A. Point Prediction
X_new = make_features(np.array([day_input]), np.array([dose_input]))
pred_point = model.predict(X_new)[0]

# B. Interval Logic
q_val = None
source_msg = ""

if pred_type == "Daily Release Rate":
    q_val, source_msg = get_rate_quantile(dose_input, confidence_level)
    
    if q_val is not None:
        # Allow negative lower bounds
        lower_bound = pred_point - q_val
        upper_bound = pred_point + q_val
    else:
        # Fallback if file missing
        st.warning(f"Could not load conformity scores for Dose {dose_input}%. Intervals disabled.")
        lower_bound = upper_bound = None
else:
    # Cumulative mode: No intervals
    lower_bound = upper_bound = None

# ------------------------------------------
# 5. Display Results
# ------------------------------------------
# Dynamic columns: 3 if we have intervals, 1 if we don't
if q_val is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Predicted Value", value=f"{pred_point:.4f} {unit}")
    with col2:
        st.metric(label="Lower Bound", value=f"{lower_bound:.4f} {unit}", delta=f"-{q_val:.4f}")
    with col3:
        st.metric(label="Upper Bound", value=f"{upper_bound:.4f} {unit}", delta=f"+{q_val:.4f}")
    st.caption(f"Interval Source: {source_msg} | Quantile: {confidence_level}")
else:
    col1 = st.columns(1)[0]
    with col1:
        st.metric(label="Predicted Value", value=f"{pred_point:.4f} {unit}")
    if pred_type == "Cumulative Release":
        st.caption("Prediction intervals are not available for Cumulative Release.")

# ------------------------------------------
# 6. Visualization
# ------------------------------------------
st.markdown("---")
st.subheader("Prediction Curve")

# Generate curve up to 663 days
t_plot = np.linspace(0, 663, 1000)
d_plot = np.full_like(t_plot, dose_input)
X_plot = make_features(t_plot, d_plot)
y_plot = model.predict(X_plot)

# Filter actual data for plotting
df_dose_plot = df_calib[df_calib["Dose"] == dose_input]

fig, ax = plt.subplots(figsize=(10, 5))

# Plot Interval (Only if q_val exists)
if q_val is not None:
    # Allow negative lower bounds in plot
    y_lower = y_plot - q_val
    y_upper = y_plot + q_val
    ax.fill_between(t_plot, y_lower, y_upper, color='purple', alpha=0.2, label=f'{int(confidence_level*100)}% Prediction Interval')

# Plot Mean Curve
ax.plot(t_plot, y_plot, color='purple', linewidth=2, label='Model Prediction')

# Plot Actual Data
if not df_dose_plot.empty:
    ax.scatter(df_dose_plot["Day"], df_dose_plot["Target"], color='black', s=30, label='Actual Data', zorder=3)

# Plot User Point
ax.scatter([day_input], [pred_point], color='red', s=100, marker='*', label='Selected Day', zorder=4)

# Error bar on user point (Only if q_val exists)
if q_val is not None:
    ax.errorbar([day_input], [pred_point], yerr=[[pred_point-lower_bound], [upper_bound-pred_point]], fmt='none', ecolor='red', capsize=5, zorder=4)

ax.set_xlabel("Days")
ax.set_ylabel(f"{pred_type} ({unit})")
ax.set_title(f"Dose {dose_input}%: {pred_type} over Time")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
