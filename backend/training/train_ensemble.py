"""
Ensemble Meta-Learner Training
Combines LightGBM and GRU OOF predictions using LogisticRegression.
"""

import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import yaml
import os
import sys

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def train_ensemble():
    print("=" * 70)
    print("  Ensemble Meta-Learner Training")
    print("=" * 70)

    with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    ens_cfg = config["ensemble"]

    # ── Load OOF predictions ──
    lgbm_oof = np.load(os.path.join(ROOT, "models", "lgbm_oof_preds.npy"))
    gru_oof = np.load(os.path.join(ROOT, "models", "gru_oof_preds.npy"))
    print(f"  LightGBM OOF: {len(lgbm_oof)} samples")
    print(f"  GRU OOF: {len(gru_oof)} samples")

    # ── Load base data for meta-features ──
    df = pd.read_csv(os.path.join(ROOT, "data", "weekly_behavioral_features.csv"))
    train_df = df[df["week_number"] <= 40].copy()

    # Align lengths with the smaller set (GRU has fewer due to sequence windowing)
    min_len = min(len(lgbm_oof), len(gru_oof))

    # For the ensemble, we need to align. LightGBM has per-row OOF,
    # GRU has per-sequence OOF. We'll use the shorter length.
    lgbm_subset = lgbm_oof[:min_len]
    gru_subset = gru_oof[:min_len]

    # Get corresponding metadata
    if min_len <= len(train_df):
        meta_df = train_df.iloc[:min_len]
    else:
        meta_df = train_df

    actual_len = min(min_len, len(meta_df))
    lgbm_subset = lgbm_subset[:actual_len]
    gru_subset = gru_subset[:actual_len]
    meta_df = meta_df.iloc[:actual_len]

    # ── Build meta-features ──
    X_meta = np.column_stack([
        lgbm_subset,
        gru_subset,
        meta_df["week_number"].values / 52.0,
        meta_df["stress_level"].values / 2.0,
        (meta_df.get("credit_utilization", pd.Series(np.zeros(actual_len))).values)
    ])
    y_meta = meta_df["will_default_next_30d"].values

    print(f"  Meta-features shape: {X_meta.shape}")
    print(f"  Positive rate: {y_meta.mean():.1%}")

    # ── Train/Val split ──
    X_trn, X_val, y_trn, y_val = train_test_split(
        X_meta, y_meta, test_size=0.2, stratify=y_meta, random_state=42)

    # ── Fit LogisticRegression ──
    meta_model = LogisticRegression(
        C=ens_cfg["C"],
        max_iter=ens_cfg["max_iter"],
        random_state=42,
        class_weight="balanced"
    )
    meta_model.fit(X_trn, y_trn)

    # ── Evaluate ──
    val_preds = meta_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"\n  Ensemble Validation ROC-AUC: {val_auc:.4f}")

    full_preds = meta_model.predict_proba(X_meta)[:, 1]
    full_auc = roc_auc_score(y_meta, full_preds)
    print(f"  Ensemble Full ROC-AUC: {full_auc:.4f}")

    # ── Save model ──
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    model_path = os.path.join(ROOT, "models", "ensemble_meta.pkl")
    joblib.dump(meta_model, model_path)
    print(f"  Saved: {model_path}")
    print(f"\n  [OK] Ensemble training complete!")
    return meta_model, val_auc


if __name__ == "__main__":
    train_ensemble()
