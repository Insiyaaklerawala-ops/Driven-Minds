import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

logger = logging.getLogger(__name__)


# ---------------------------
# CLEAN TARGET
# ---------------------------
def clean_target(y: pd.Series):
    """Normalize a raw label column into binary 0/1 + a validity mask."""
    y = y.astype(str).str.strip()

    mapping = {
        '>50K': 1, '<=50K': 0,
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0
    }

    y = y.map(mapping)
    valid = y.notna()
    return y[valid].astype(int), valid


# ---------------------------
# ENCODE FEATURES
# ---------------------------
def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns and fill missing values."""
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(0)


# ---------------------------
# ANALYZE BIAS (STRONG MODEL)
# ---------------------------
def analyze_bias(df: pd.DataFrame, label_col: str, sensitive_col: str) -> dict:
    """
    Train a classifier on the given dataframe and measure demographic
    parity / equalized odds bias with respect to `sensitive_col`.
    """
    df = df.copy()

    # Target
    y_raw = df[label_col]
    y, valid_idx = clean_target(y_raw)
    df = df.loc[valid_idx]

    # Features
    sensitive = df[sensitive_col].astype(str).fillna("Unknown")
    X = df.drop(columns=[label_col, sensitive_col])
    X = encode_features(X)

    if len(X) < 20:
        raise ValueError("Dataset too small after cleaning (fewer than 20 valid rows)")

    logger.info("analyze_bias: %d rows after cleaning", len(X))

    # Split
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if len(set(y_te)) < 2 or len(set(y_pred)) < 2:
        return {"note": "Only one class predicted"}

    # Metrics
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

    # Group rates
    group_rates = {}
    for g in s_te.unique():
        mask = s_te == g
        group_rates[str(g)] = round(float(y_pred[mask].mean()), 3)

    return {
        "accuracy": round(acc * 100, 1),
        "bias_score": round(abs(dpd), 3),
        "raw_dpd": round(dpd, 3),
        "equalized_odds_diff": round(eod, 3),
        "groups": list(s_te.unique()),
        "group_rates": group_rates,
        "is_biased": abs(dpd) > 0.1,
        "sensitive_col": sensitive_col
    }


# ---------------------------
# MITIGATE BIAS (FAIR MODEL, OPTIMIZED)
# ---------------------------
def mitigate_bias(df: pd.DataFrame, label_col: str, sensitive_col: str) -> dict:
    """
    Retrain using fairlearn's ExponentiatedGradient reduction to reduce
    demographic parity difference, and report the improved metrics.
    """
    df = df.copy()

    y_raw = df[label_col]
    y, valid_idx = clean_target(y_raw)
    df = df.loc[valid_idx]

    sensitive = df[sensitive_col]
    X = df.drop(columns=[label_col, sensitive_col])
    X = encode_features(X)

    # Cap dataset size for speed
    if len(X) > 5000:
        X = X.sample(5000, random_state=42)
        y = y.loc[X.index]
        sensitive = sensitive.loc[X.index]

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )

    base_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=6,
        random_state=42
    )

    fair_model = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity(),
        max_iter=10
    )

    logger.info("mitigate_bias: starting fairness-constrained training")
    fair_model.fit(X_tr, y_tr, sensitive_features=s_tr)
    y_pred_fair = fair_model.predict(X_te)

    new_dpd = demographic_parity_difference(
        y_te, y_pred_fair, sensitive_features=s_te
    )
    new_acc = accuracy_score(y_te, y_pred_fair)

    return {
        "after_bias_score": round(abs(new_dpd), 3),
        "after_accuracy": round(new_acc * 100, 1),
        "is_fixed": abs(new_dpd) <= 0.1
    }