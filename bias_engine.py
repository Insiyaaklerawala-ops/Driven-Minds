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
    # 🔥 CLEAN TARGET
    # ---------------------------
    y = df[label_col].astype(str).str.strip()
    y = y.str.replace('.', '', regex=False)

    y = y.map({
        '>50K': 1,
        '<=50K': 0
    })

    # Remove invalid rows
    valid_idx = y.notna()
    df = df[valid_idx]
    y = y[valid_idx].astype(int)

    # ---------------------------
    # 🔥 CLEAN SENSITIVE FEATURE
    # ---------------------------
    sensitive = df[sensitive_col].astype(str).str.strip()

    # ---------------------------
    # FEATURES
    # ---------------------------
    X = df.drop(columns=[label_col, sensitive_col])

    # Fill missing values
    X = X.fillna("missing")

    # Encode categorical columns
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # ---------------------------
    # SPLIT
    # ---------------------------
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )

    # ---------------------------
    # 🔥 MODEL (NO WARNINGS)
    # ---------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ---------------------------
    # SAFETY CHECK
    # ---------------------------
    if len(set(y_te)) < 2 or len(set(y_pred)) < 2:
        return {
            "error": "Only one class predicted — cannot compute fairness metrics."
        }

    # ---------------------------
    # METRICS
    # ---------------------------
    acc = accuracy_score(y_te, y_pred)

    dpd = demographic_parity_difference(
        y_te, y_pred, sensitive_features=s_te
    )

    eod = equalized_odds_difference(
        y_te, y_pred, sensitive_features=s_te
    )

    return {
        "accuracy": float(round(acc * 100, 1)),
        "demographic_parity_diff": float(round(dpd, 3)),
        "equalized_odds_diff": float(round(eod, 3)),
        "bias_score": float(round(abs(dpd), 3)),
        "is_biased": bool(abs(dpd) > 0.1),
        "sensitive_col": sensitive_col,
        "groups": list(s_te.unique())
    }


# ---------------------------
# 🔥 LOAD DATA SAFELY
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "adult.csv")

df = pd.read_csv(file_path)


# ---------------------------
# 🚀 RUN
# ---------------------------
print("Gender Bias:")
print(analyze_bias(df, "income", "gender"))

print("\nRace Bias:")
print(analyze_bias(df, "income", "race"))