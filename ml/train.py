# ml/train.py
# Trains a Logistic Regression baseline and an XGBoost model.
# Prints a side-by-side comparison table and saves:
#   models/model.pkl     – the XGBoost model
#   models/metrics.json  – all evaluation metrics from this run
#
# Run once: python ml/train.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier

from ml.features import build_features, FEATURE_COLUMNS

DATA_PATH = os.path.join("data", "clients_monthly.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def top20_capture(y_true, y_prob) -> float:
    """
    Share of true churners among the top 20% highest-risk clients.
    This measures how useful the model is for triage.
    """
    df = pd.DataFrame({"true": y_true, "prob": y_prob})
    df = df.sort_values("prob", ascending=False)
    top20 = df.head(max(1, int(len(df) * 0.20)))
    if y_true.sum() == 0:
        return 0.0
    return top20["true"].sum() / y_true.sum()


def evaluate(name, y_true, y_prob, threshold=0.5) -> dict:
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "top20_capture": round(top20_capture(pd.Series(y_true), pd.Series(y_prob)), 4),
    }


def main():
    print("Loading data...")
    raw = pd.read_csv(DATA_PATH)
    print(f"  {len(raw)} rows, {raw['client_id'].nunique()} clients")
    print(f"  Overall churn rate: {raw.groupby('client_id')['churned'].max().mean():.3f}")

    # Attach label (churned) per client: 1 if churned in any month
    churn_label = raw.groupby("client_id")["churned"].max().reset_index()
    churn_label.columns = ["client_id", "label"]

    # Build features (drop churned column first so build_features doesn't see it)
    data_no_label = raw.drop(columns=["churned"])
    features_df, _ = build_features(data_no_label)

    # Merge label
    features_df = features_df.merge(churn_label, on="client_id")
    X = features_df[FEATURE_COLUMNS].values
    y = features_df["label"].values
    print(f"  Feature matrix: {X.shape}, churn rate in features: {y.mean():.3f}")

    # Stratified 75/25 split by client
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} clients, Test: {X_test.shape[0]} clients")

    # --- Baseline: Logistic Regression ---
    print("\nTraining Logistic Regression baseline...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate("LogisticRegression", y_test, lr_prob)

    # --- Main model: XGBoost ---
    print("Training XGBoost...")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight = {scale_pos_weight:.2f} (negatives/positives)")

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate("XGBoost", y_test, xgb_prob)

    # --- Print comparison table ---
    print("\n" + "=" * 65)
    print(f"{'Metric':<20} {'LogisticRegression':>20} {'XGBoost':>20}")
    print("-" * 65)
    for key in ["auc", "recall", "precision", "f1", "top20_capture"]:
        print(f"{key:<20} {lr_metrics[key]:>20.4f} {xgb_metrics[key]:>20.4f}")
    print("=" * 65)

    # --- Save model and metrics ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    all_metrics = {
        "xgboost": xgb_metrics,
        "logistic_regression": lr_metrics,
        "training_info": {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "churn_rate_overall": round(float(y.mean()), 4),
            "scale_pos_weight": round(float(scale_pos_weight), 4),
            "feature_columns": FEATURE_COLUMNS,
        },
        "disclaimer": (
            "Trained on synthetic data. The churn label is planted by the generator. "
            "Risk % is a ranking score, not a calibrated probability."
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
