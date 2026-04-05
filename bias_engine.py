import pandas as pd 
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
def analyze_bias(df, label_col, sensitive_col):
    df = df.copy()

    # ---------------------------
    # ✅ TARGET CLEANING (GENERIC)
    # ---------------------------
    y = pd.to_numeric(df[label_col], errors='coerce')

    # Drop invalid
    valid_idx = y.notna()
    df = df.loc[valid_idx].copy()
    y = y.loc[valid_idx].astype(int)

    # Ensure binary
    y = y[y.isin([0, 1])]
    df = df.loc[y.index]

    if len(df) < 30:
        print("⚠️ Warning: dataset small after cleaning")

    # ---------------------------
    # ✅ SENSITIVE FEATURE
    # ---------------------------
    sensitive = df[sensitive_col].astype(str).fillna("Unknown")

    # ---------------------------
    # ✅ FEATURES
    # ---------------------------
    X = df.drop(columns=[label_col, sensitive_col]).copy()

    # Keep only numeric
    X = X.select_dtypes(include=['number'])

    # Fill numeric safely
    X = X.fillna(0)

    if len(X) < 10:
        raise ValueError("Not enough data after preprocessing")

    # ---------------------------
    # ✅ SPLIT
    # ---------------------------
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )

    # ---------------------------
    # ✅ MODEL
    # ---------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ---------------------------
    # ✅ SAFETY CHECK
    # ---------------------------
    if len(set(y_te)) < 2 or len(set(y_pred)) < 2:
        return {
            "accuracy": 0,
            "bias_score": 0,
            "is_biased": False,
            "groups": list(s_te.unique()),
            "sensitive_col": sensitive_col,
            "note": "Only one class predicted"
        }

    # ---------------------------
    # ✅ METRICS
    # ---------------------------
    acc = accuracy_score(y_te, y_pred)

    dpd = demographic_parity_difference(
        y_true=y_te,
        y_pred=y_pred,
        sensitive_features=s_te
    )

    eod = equalized_odds_difference(
        y_true=y_te,
        y_pred=y_pred,
        sensitive_features=s_te
    )

    # ---------------------------
    # ✅ OUTPUT
    # ---------------------------
    return {
        "accuracy": float(round(acc * 100, 1)),
        "demographic_parity_diff": float(round(dpd, 3)),
        "equalized_odds_diff": float(round(eod, 3)),
        "bias_score": float(round(abs(dpd), 3)),
        "is_biased": bool(abs(dpd) > 0.1),
        "sensitive_col": sensitive_col,
        "groups": sorted(list(s_te.unique()))
    }