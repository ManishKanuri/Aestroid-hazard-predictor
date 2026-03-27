import joblib
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# dist_min is intentionally excluded — it is used to define the label.
# dist (nominal close approach) is kept: dist != dist_min so no leakage.
FEATURES = ["dist", "v_rel", "v_inf", "h", "diameter_avg_km"]
MODEL_PATH = Path(__file__).parent.parent / "models" / "asteroid_model.pkl"


def train(df: pd.DataFrame) -> Pipeline:
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    y = df["hazardous"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train class distribution — Safe: {(y_train==0).sum()}  Hazardous: {(y_train==1).sum()}")

    # SMOTE on training split only (never on test — that would be leakage)
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE          — Safe: {(y_res==0).sum()}  Hazardous: {(y_res==1).sum()}")

    scaler = StandardScaler()
    X_res_sc  = scaler.fit_transform(X_res)
    X_test_sc = scaler.transform(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    clf.fit(X_res_sc, y_res)

    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Hazardous"]))

    joblib.dump({"scaler": scaler, "clf": clf, "features": available}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return scaler, clf


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["scaler"], artifact["clf"], artifact["features"]
