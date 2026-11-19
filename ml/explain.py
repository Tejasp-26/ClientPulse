# ml/explain.py
# Model loading, scoring, and SHAP-based explanation.
# load_model()       – loads model.pkl from disk
# score_clients()    – adds risk and risk_level columns
# get_reasons()      – returns top-N real warning signs using SHAP + rules
# get_shap_values()  – returns raw SHAP values for one client (for the detail chart)

import os
import numpy as np
import pandas as pd
import joblib
import shap

from ml.features import FEATURE_COLUMNS
from ml.actions import is_real_problem, reason_text

MODEL_PATH = os.path.join("models", "model.pkl")


def load_model():
    """Load the trained XGBoost model from models/model.pkl."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python ml/train.py"
        )
    return joblib.load(MODEL_PATH)


def score_clients(
    model,
    features_df: pd.DataFrame,
    high: float = 0.70,
    medium: float = 0.40,
) -> pd.DataFrame:
    """
    Add 'risk' (0-1) and 'risk_level' (High/Medium/Low) columns to features_df.
    If medium >= high, adjust medium to high - 0.01 and log a warning.
    """
    if medium >= high:
        medium = high - 0.01
        print(f"Warning: medium threshold adjusted to {medium:.2f} (must be < high={high:.2f})")

    X = features_df[FEATURE_COLUMNS].values
    risk = model.predict_proba(X)[:, 1]

    def bucket(r):
        if r >= high:
            return "High"
        elif r >= medium:
            return "Medium"
        return "Low"

    result = features_df[["client_id"]].copy()
    result["risk"] = risk
    result["risk_level"] = [bucket(r) for r in risk]
    return result


def get_reasons(
    model,
    features_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Use SHAP to find the top features driving risk for each client.
    Only features that pass is_real_problem() are kept.
    Returns a DataFrame with columns: client_id, top_reasons_raw (list of dicts).
    """
    X = features_df[FEATURE_COLUMNS].values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)   # shape (n_clients, n_features)

    # For binary classifiers shap may return a list; take class-1 values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    records = []
    for i, client_id in enumerate(features_df["client_id"]):
        client_shap = shap_values[i]         # one value per feature
        client_vals = X[i]

        # Pair features with their SHAP values, keep positive SHAP only
        pairs = [
            (FEATURE_COLUMNS[j], float(client_vals[j]), float(client_shap[j]))
            for j in range(len(FEATURE_COLUMNS))
            if client_shap[j] > 0
        ]
        # Sort biggest SHAP first
        pairs.sort(key=lambda x: x[2], reverse=True)

        # Keep only real problems (rules filter)
        reasons = []
        for feat, val, shap_val in pairs:
            if is_real_problem(feat, val):
                reasons.append({
                    "feature": feat,
                    "value": val,
                    "shap": shap_val,
                    "text": reason_text(feat, val),
                })
            if len(reasons) >= top_n:
                break

        records.append({"client_id": client_id, "top_reasons_raw": reasons})

    return pd.DataFrame(records)


def get_shap_values(model, client_features_df: pd.DataFrame) -> list[float]:
    """
    Return raw SHAP values for a single client row.
    Used by the detail page to draw the SHAP bar chart.
    """
    X = client_features_df[FEATURE_COLUMNS].values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # Return as a flat list (one value per feature)
    return shap_values[0].tolist()
