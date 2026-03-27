import joblib
import numpy as np
import streamlit as st

MODEL_PATH = "models/asteroid_model.pkl"

st.set_page_config(page_title="Asteroid Hazard Predictor", page_icon="☄️", layout="centered")

st.title("☄️ Asteroid Hazard Predictor")
st.markdown("Enter orbital parameters to predict whether an asteroid poses a hazard to Earth.")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    dist = st.number_input(
        "Distance (AU)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%.4f",
        help="Close approach distance in Astronomical Units",
    )
with col2:
    v_rel = st.number_input(
        "Relative Velocity (km/s)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1,
        help="Velocity relative to Earth at close approach",
    )
with col3:
    h = st.number_input(
        "Absolute Magnitude (H)",
        min_value=0.0,
        max_value=35.0,
        value=22.0,
        step=0.1,
        help="Lower H = larger asteroid (H < 22 ≈ diameter > 140m)",
    )

st.divider()

if st.button("Predict Hazard", type="primary", use_container_width=True):
    try:
        model = joblib.load(MODEL_PATH)
        features = np.array([[dist, v_rel, h]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        hazard_prob = probability[1] * 100
        safe_prob = probability[0] * 100

        if prediction == 1:
            st.error(f"**HAZARDOUS** — {hazard_prob:.1f}% hazard probability")
            st.progress(hazard_prob / 100)
        else:
            st.success(f"**SAFE** — {safe_prob:.1f}% safe probability")
            st.progress(safe_prob / 100)

        st.markdown("#### Probability Breakdown")
        col_s, col_h = st.columns(2)
        col_s.metric("Safe", f"{safe_prob:.1f}%")
        col_h.metric("Hazardous", f"{hazard_prob:.1f}%")

    except FileNotFoundError:
        st.warning("Model not found. Please run `python main.py` first to train the model.")
