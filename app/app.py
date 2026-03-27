import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import fetch_neows_live, fetch_sentry_data, fetch_cad_data
from src.preprocessing import preprocess
from src.model import MODEL_PATH, train

# ── Auto-train if model is missing or was built on a different Python version ─
def ensure_model():
    needs_train = False
    if not Path(MODEL_PATH).exists():
        needs_train = True
    else:
        try:
            import joblib
            joblib.load(MODEL_PATH)
        except Exception:
            needs_train = True

    if needs_train:
        with st.spinner("First run: training model on NASA JPL data (~60 seconds)..."):
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            df_raw = fetch_cad_data(date_min="2015-01-01", date_max="2024-01-01")
            df_clean = preprocess(df_raw)
            train(df_clean)

ensure_model()

st.set_page_config(
    page_title="Space Intelligence Platform",
    page_icon="☄️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; color: #e0e0e0; }
    .metric-card { background: #111133; border: 1px solid #334; border-radius: 10px; padding: 16px; }
    h1, h2, h3 { color: #00d4ff !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
    .byline { font-size: 13px; color: #556; margin-top: -8px; }
    .byline a { color: #00d4ff; text-decoration: none; }
    .byline a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.title("☄️ Space Intelligence Platform")
col_title, col_by = st.columns([6, 1])
with col_title:
    st.caption("Real-time asteroid hazard analysis powered by NASA NeoWs · JPL Sentry · CelesTrak")
with col_by:
    st.markdown('<p class="byline">by <a href="https://github.com/ManishKanuri" target="_blank">Manish Kanuri</a></p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("NASA API Key", value="DEMO_KEY",
                            help="Get a free key at api.nasa.gov (higher rate limits)")
    days = st.slider("NEO feed window (days)", 1, 7, 7)
    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- [NASA NeoWs API](https://api.nasa.gov)")
    st.markdown("- [JPL Sentry](https://cneos.jpl.nasa.gov/sentry)")
    st.markdown("- [CelesTrak TLE](https://celestrak.org)")
    st.divider()
    st.markdown("Built by **Manish Kanuri**")
    st.markdown("[GitHub](https://github.com/ManishKanuri/Aestroid-hazard-predictor)")

# ── Load live data ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_neo_data(days, api_key):
    return fetch_neows_live(days=days, api_key=api_key)

@st.cache_data(ttl=3600, show_spinner=False)
def load_sentry():
    return fetch_sentry_data()

with st.spinner("Fetching live NASA data..."):
    try:
        df_raw = load_neo_data(days, api_key)
        df = preprocess(df_raw)
        neo_ok = True
    except Exception as e:
        st.error(f"NeoWs API error: {e}")
        df = pd.DataFrame()
        neo_ok = False

    try:
        df_sentry = load_sentry()
        sentry_ok = True
    except Exception as e:
        df_sentry = pd.DataFrame()
        sentry_ok = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📡 Live NEO Feed", "💥 Impact Risk (Sentry)", "🌍 3D Tracker"])

# ── TAB 1: Predictor ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Asteroid Hazard Predictor")
    st.markdown("Enter orbital parameters to predict hazard probability.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dist = st.number_input("Distance (AU)", 0.0, 1.0, 0.05, 0.001, format="%.4f",
                               help="Close approach distance in Astronomical Units")
    with col2:
        v_rel = st.number_input("Velocity (km/s)", 0.0, 100.0, 15.0, 0.1)
    with col3:
        h_mag = st.number_input("Abs. Magnitude (H)", 0.0, 35.0, 22.0, 0.1,
                                help="Lower = larger asteroid. H<22 ≈ diameter >140m")
    with col4:
        diameter = st.number_input("Diameter (km, approx)", 0.0, 50.0, 0.14, 0.01)

    if st.button("Predict", type="primary", use_container_width=True):
        try:
            from src.model import load_model
            scaler, clf, features = load_model()

            diameter_avg = diameter if diameter > 0 else 10 ** (3.1236 - 0.5 * h_mag) / 1000
            v_inf = v_rel * 0.9

            input_map = {
                "dist": dist, "v_rel": v_rel, "v_inf": v_inf,
                "h": h_mag, "diameter_avg_km": diameter_avg,
            }
            X = pd.DataFrame([[input_map[f] for f in features]], columns=features)
            X_scaled = scaler.transform(X)

            pred = clf.predict(X_scaled)[0]
            proba = clf.predict_proba(X_scaled)[0]
            hazard_pct = proba[1] * 100
            safe_pct = proba[0] * 100

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Hazard Probability", f"{hazard_pct:.1f}%")
            c2.metric("Safe Probability", f"{safe_pct:.1f}%")
            c3.metric("Verdict", "⚠️ HAZARDOUS" if pred == 1 else "✅ SAFE")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hazard_pct,
                number={"suffix": "%"},
                title={"text": "Hazard Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4444" if pred == 1 else "#44ff88"},
                    "steps": [
                        {"range": [0, 30], "color": "#0a2a0a"},
                        {"range": [30, 70], "color": "#2a2a0a"},
                        {"range": [70, 100], "color": "#2a0a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
                },
            ))
            fig.update_layout(paper_bgcolor="#0a0a1a", font_color="#e0e0e0", height=300)
            st.plotly_chart(fig, use_container_width=True)

        except FileNotFoundError:
            st.warning("Model not found. Run `python main.py` first to train.")

# ── TAB 2: Live NEO Feed ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Live NEO Feed — Last {days} Days")

    if neo_ok and not df.empty:
        h_count = int(df["hazardous"].sum())
        s_count = len(df) - h_count

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total NEOs", len(df))
        m2.metric("Hazardous", h_count, delta=None)
        m3.metric("Safe", s_count)
        m4.metric("Avg Velocity", f"{df['v_rel'].mean():.1f} km/s")

        # Scatter: Distance vs Velocity
        fig_scatter = px.scatter(
            df, x="dist", y="v_rel", color="hazardous",
            color_discrete_map={0: "#44ff88", 1: "#ff4444"},
            size="diameter_avg_km", hover_data=["name", "h", "close_approach_date"],
            labels={"dist": "Distance (AU)", "v_rel": "Velocity (km/s)", "hazardous": "Hazardous"},
            title="Close Approach Distance vs Velocity",
            template="plotly_dark",
        )
        fig_scatter.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0d0d2a")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Histogram: H magnitude distribution
        col_a, col_b = st.columns(2)
        with col_a:
            fig_h = px.histogram(df, x="h", color="hazardous",
                                 color_discrete_map={0: "#44ff88", 1: "#ff4444"},
                                 title="Absolute Magnitude Distribution",
                                 template="plotly_dark", nbins=30)
            fig_h.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0d0d2a")
            st.plotly_chart(fig_h, use_container_width=True)

        with col_b:
            pie_data = pd.DataFrame({"Status": ["Safe", "Hazardous"], "Count": [s_count, h_count]})
            fig_pie = px.pie(pie_data, values="Count", names="Status",
                             color_discrete_sequence=["#44ff88", "#ff4444"],
                             title="Hazard Breakdown", template="plotly_dark")
            fig_pie.update_layout(paper_bgcolor="#0a0a1a")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Data table
        st.markdown("#### NEO Details")
        display_cols = ["name", "close_approach_date", "dist", "v_rel", "h", "diameter_avg_km", "hazardous"]
        available_cols = [c for c in display_cols if c in df.columns]
        styled = df[available_cols].sort_values("dist").reset_index(drop=True)
        st.dataframe(styled, use_container_width=True, height=350)
    else:
        st.info("Could not load NEO data. Check your API key or network connection.")

# ── TAB 3: Sentry Impact Risk ─────────────────────────────────────────────────
with tab3:
    st.subheader("JPL Sentry — Potential Earth Impactors")
    st.caption("Objects with non-zero probability of Earth impact in the next 100 years.")

    if sentry_ok and not df_sentry.empty:
        top = df_sentry.copy()
        if "ip" in top.columns:
            top = top.sort_values("ip", ascending=False).head(50)

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects Monitored", len(df_sentry))
        if "ip" in df_sentry.columns:
            m2.metric("Highest Impact Prob.", f"{df_sentry['ip'].max():.2e}")
        if "ts_max" in df_sentry.columns:
            m3.metric("Max Torino Scale", int(df_sentry["ts_max"].max()))

        # Impact probability chart
        if "ip" in top.columns and "des" in top.columns:
            fig_bar = px.bar(
                top.head(20), x="des", y="ip",
                color="ip", color_continuous_scale="reds",
                title="Top 20 Objects by Impact Probability",
                labels={"des": "Object", "ip": "Impact Probability"},
                template="plotly_dark",
            )
            fig_bar.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#0d0d2a",
                                  xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        show_cols = [c for c in ["des", "fullname", "ip", "ps_max", "ts_max", "diameter", "h", "last_obs"] if c in top.columns]
        st.dataframe(top[show_cols].reset_index(drop=True), use_container_width=True, height=400)
    else:
        st.info("Could not load Sentry data.")

# ── TAB 4: 3D Tracker Link ────────────────────────────────────────────────────
with tab4:
    st.subheader("3D Earth Tracker")
    st.caption("Live satellites (CelesTrak TLE · SGP4) + NASA NeoWs asteroids — drag to rotate, scroll to zoom, hover for details")
    with open(Path(__file__).parent / "3d-visualization" / "index.html", "r") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=700, scrolling=False)
