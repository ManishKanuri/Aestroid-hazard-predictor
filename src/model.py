import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["dist", "v_rel", "h", "diameter_avg_km", "threat_score"]
MODEL_PATH = "models/asteroid_model.pkl"


def train(df: pd.DataFrame) -> Pipeline:
    """Train a GradientBoosting pipeline to predict hazardous asteroids."""
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]
    y = df["hazardous"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Hazardous"]))

    joblib.dump({"model": pipeline, "features": available}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return pipeline


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["features"]
