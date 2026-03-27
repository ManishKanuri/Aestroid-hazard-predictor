# ☄️ Asteroid Hazard Predictor

A machine learning platform that predicts whether asteroids pose a hazard to Earth, using real NASA JPL close-approach data — with a 3D interactive Earth visualization.

---

## Features

- **Live NASA Data** — fetches real asteroid close-approach data from the NASA JPL API
- **ML Hazard Prediction** — RandomForest classifier trained on distance, velocity, and magnitude
- **Streamlit Web App** — interactive UI to predict hazard probability for any asteroid
- **3D Earth Visualization** — real-time orbiting asteroids rendered with Three.js

---

## Project Structure

```
Aestroid/
├── data/                          # Raw data storage
├── models/                        # Saved trained model
├── src/
│   ├── data_loader.py             # NASA JPL API → DataFrame
│   ├── preprocessing.py           # Cleaning + feature engineering
│   └── model.py                   # RandomForest training + evaluation
├── app/
│   ├── app.py                     # Streamlit prediction app
│   └── 3d-visualization/
│       └── index.html             # 3D Earth + Asteroids (Three.js)
├── main.py                        # Full pipeline runner
└── requirements.txt
```

---

## Quickstart

### 1. Clone the repo
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
This fetches NASA data, preprocesses it, trains the model, and saves it to `models/asteroid_model.pkl`.

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`

### 5. Open 3D Visualization
```bash
open app/3d-visualization/index.html
```
Or just drag the file into any browser (Chrome recommended).

---

## How It Works

### Data
Fetched from the [NASA JPL Close Approach API](https://ssd-api.jpl.nasa.gov/cad.api):
- `dist` — close approach distance (AU)
- `v_rel` — relative velocity (km/s)
- `h` — absolute magnitude (proxy for size)

### Hazard Label
An asteroid is labeled **hazardous** if:
```
dist < 0.05 AU  AND  H < 22
```
This aligns with NASA's Potentially Hazardous Asteroid (PHA) definition.

### Model
- Algorithm: `RandomForestClassifier` (scikit-learn)
- Features: `dist`, `v_rel`, `h`
- Output: binary classification + hazard probability

---

## 3D Visualization

Built with [Three.js](https://threejs.org/):

| Color | Meaning |
|-------|---------|
| 🔴 Red | Hazardous (dist < 0.05 AU, H < 22) |
| 🟠 Orange | Near (dist < 0.1 AU) |
| 🟢 Green | Safe |
| 🔵 Blue | Earth |

- Hover over any asteroid for name, distance, velocity, and hazard status
- Drag to rotate, scroll to zoom
- Auto-falls back to demo data if API is unreachable

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Data | NASA JPL API, pandas |
| ML | scikit-learn, joblib |
| App | Streamlit |
| 3D | Three.js |
| Language | Python 3.8+, JavaScript |

---

## License

MIT
