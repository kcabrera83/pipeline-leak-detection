import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Pipeline Leak Detection", layout="wide")
st.title("Pipeline Leak Detection")
st.markdown("Detect leaks in oil & gas pipelines using ensemble methods.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("leak", "leak_classifier.pkl"), ("size", "leak_size_estimator.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
upstream_pressure_psi = st.sidebar.slider("Upstream Pressure Psi", 100, 1500, 800)
downstream_pressure_psi = st.sidebar.slider("Downstream Pressure Psi", 50, 1400, 725)
flow_rate_gpm = st.sidebar.slider("Flow Rate Gpm", 0, 500, 250)
acoustic_emission_db = st.sidebar.slider("Acoustic Emission Db", 0, 100, 50)
vibration_mm_s = st.sidebar.slider("Vibration Mm S", 0, 50, 25)
temperature_diff_c = st.sidebar.slider("Temperature Diff C", 0, 30, 15)
soil_humidity_pct = st.sidebar.slider("Soil Humidity Pct", 0, 100, 50)
pipe_type = st.sidebar.selectbox("Pipe Type", ['steel', 'plastic', 'composite'])

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[upstream_pressure_psi, downstream_pressure_psi, flow_rate_gpm, acoustic_emission_db, vibration_mm_s, temperature_diff_c, soil_humidity_pct, pipe_type]])
        m = models["leak"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Leak", result if isinstance(result, str) else f"{result:.4f}")
        m = models["size"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Size", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

