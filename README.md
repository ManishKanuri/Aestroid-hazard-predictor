# ☄️ Asteroid Hazard Predictor

> A production-grade machine learning platform for real-time asteroid threat analysis, powered by NASA & JPL live data feeds and rendered in an interactive 3D Earth visualization.

---

## Overview

The Asteroid Hazard Predictor ingests live data from multiple space agency APIs, trains a gradient boosting classifier with rigorous anti-leakage design, and serves predictions through a multi-tab Streamlit dashboard — alongside a Three.js 3D visualization of real satellites and near-Earth objects.

---

## Features

| Feature | Details |
|---------|---------|
| **Live NASA Data** | NeoWs daily NEO feed + JPL Sentry impact risk |
| **Bulk Training Data** | JPL CAD API — 50,000+ historical close approaches, no rate limits |
| **ML Pipeline** | GradientBoosting + SMOTE, SMOTE applied to training split only |
| **No Data Leakage** | Label uses `dist_min` (3σ uncertainty bound); feature uses `dist` (nominal) |
| **Streamlit Dashboard** | 4-tab app: Predictor, Live NEO Feed, Impact Risk, 3D Tracker |
| **3D Visualization** | Three.js Earth + satellite.js SGP4 real satellite propagation |
| **Satellite Tracking** | Live CelesTrak TLE data — LEO / MEO / GEO orbit classification |

---

## Project Structure

```
Aestroid/
├── data/                          # Raw data storage
├── models/
│   └── asteroid_model.pkl         # Trained model artifact (scaler + clf + features)
├── src/
│   ├── data_loader.py             # NASA NeoWs · JPL CAD · JPL Sentry API clients
│   ├── preprocessing.py           # Feature engineering + leakage-free label design
│   └── model.py                   # GradientBoosting + SMOTE training pipeline
├── app/
│   ├── app.py                     # 4-tab Streamlit dashboard
│   └── 3d-visualization/
│       └── index.html             # Three.js + satellite.js 3D Earth tracker
├── main.py                        # End-to-end training pipeline
└── requirements.txt
```

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/ManishKanuri/Aestroid-hazard-predictor.git
cd Aestroid-hazard-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python main.py
```
Fetches 28,000+ asteroid close-approach records from JPL CAD API (no API key required), preprocesses, applies SMOTE, and trains a GradientBoosting classifier.

**Expected output:**
```
Total CAD records fetched: ~28,000
After SMOTE — Safe: 22,909  Hazardous: 22,909
Test Accuracy : 0.9995
Test ROC-AUC  : 0.9817
Hazardous  precision: 0.98  recall: 0.96  f1: 0.97
```

### 4. Launch the dashboard
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`

### 5. Open 3D Visualization
```bash
open app/3d-visualization/index.html
```

---

## Data Sources

| Source | API | Used For |
|--------|-----|---------|
| [NASA NeoWs](https://api.nasa.gov/neo) | `api.nasa.gov/neo/rest/v1/feed` | Live 7-day NEO feed, NASA PHA flags |
| [JPL CAD](https://ssd-api.jpl.nasa.gov/cad.api) | `ssd-api.jpl.nasa.gov/cad.api` | Bulk historical training data |
| [JPL Sentry](https://cneos.jpl.nasa.gov/sentry) | `ssd-api.jpl.nasa.gov/sentry.api` | Long-term impact probability scores |
| [CelesTrak TLE](https://celestrak.org) | `celestrak.org/pub/TLE/` | Live satellite orbital elements |

---

## ML Design

### Label (no leakage)
The hazard label uses `dist_min` — the 3-sigma minimum uncertainty distance from JPL CAD — while the model feature uses `dist` (nominal closest approach). These are correlated but not equal, creating a genuine predictive task.

```
hazardous = 1  if  dist_min < 0.05 AU  AND  H < 22
```

### Features
| Feature | Description |
|---------|-------------|
| `dist` | Nominal close-approach distance (AU) |
| `v_rel` | Relative velocity at closest approach (km/s) |
| `v_inf` | Hyperbolic excess velocity (km/s) |
| `h` | Absolute magnitude (proxy for size) |
| `diameter_avg_km` | Estimated diameter via Bowell formula |

### Class Imbalance
SMOTE (Synthetic Minority Oversampling) is applied **to the training split only** — the test set retains natural class distribution (1:104 imbalance) to give honest evaluation metrics.

---

## 3D Visualization

Built with [Three.js](https://threejs.org/) and [satellite.js](https://github.com/shashwatak/satellite-js):

**Satellite Mode** (default)
- Fetches live TLE data from CelesTrak
- SGP4 propagation updated every second
- Orbit trails per satellite
- Color by altitude: 🟢 LEO · 🟠 MEO · 🔴 GEO/HEO

**Asteroid Mode**
- Live NEO data from NASA NeoWs API
- Color by hazard: 🔴 Hazardous · 🟠 Near · 🟢 Safe
- Hover tooltips: name, distance, velocity, magnitude

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Predictor** | Enter orbital parameters → hazard probability gauge |
| **Live NEO Feed** | Real-time scatter/histogram/pie charts from NeoWs |
| **Impact Risk** | JPL Sentry table sorted by impact probability |
| **3D Tracker** | Link to open the Three.js visualization |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+, JavaScript (ES Modules) |
| Data | NASA NeoWs API, JPL CAD/Sentry APIs, CelesTrak TLE |
| ML | scikit-learn, imbalanced-learn (SMOTE), joblib |
| Dashboard | Streamlit, Plotly |
| 3D Engine | Three.js, satellite.js (SGP4) |

---

## License

MIT
