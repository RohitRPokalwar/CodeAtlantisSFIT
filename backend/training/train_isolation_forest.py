"""
Isolation Forest Training Script
Anomaly detection for extreme behavioral outliers.
"""

import random
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import yaml
import os
import sys

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def train_isolation_forest():
    print("=" * 70)
    print("  Isolation Forest Training - Anomaly Detection")
    print("=" * 70)

    with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    iso_cfg = config["isolation_forest"]

    ANOMALY_FEATURES = [
        "atm_withdrawal_amount_7d",
        "lending_upi_amount_7d",
        "net_cashflow_7d",
        "credit_utilization",
        "failed_autodebit_count"
    ]

    # ── Load Data ──
    df = pd.read_csv(os.path.join(ROOT, "data", "weekly_behavioral_features.csv"))
    
    # Explicit Segment Mapping (Bank-Grade Consistency)
    SEGMENT_MAP = {"salaried": 0, "self-employed": 1, "farmer": 2, "freelancer": 3, "student": 4, "other": 5}
    if "customer_segment" in df.columns:
        df["customer_segment"] = df["customer_segment"].map(SEGMENT_MAP).fillna(5).astype(int)
    
    print(f"  Loaded {len(df)} rows")
    X = df[ANOMALY_FEATURES].values
    print(f"  Features: {ANOMALY_FEATURES}")
    print(f"  Data shape: {X.shape}")

    # ── Train IsolationForest ──
    iso_model = IsolationForest(
        n_estimators=iso_cfg["n_estimators"],
        contamination=iso_cfg["contamination"],
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X)

    # ── Predict anomalies ──
    anomaly_labels = iso_model.predict(X)
    anomaly_flags = (anomaly_labels == -1).astype(int)
    anomaly_count = anomaly_flags.sum()
    print(f"  Anomalies detected: {anomaly_count} ({anomaly_count/len(X)*100:.1f}%)")

    # ── Save model ──
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    model_path = os.path.join(ROOT, "models", "isolation_forest.pkl")
    joblib.dump(iso_model, model_path)
    print(f"  Saved: {model_path}")
    print(f"\n  [OK] Isolation Forest training complete!")
    return iso_model


if __name__ == "__main__":
    train_isolation_forest()
