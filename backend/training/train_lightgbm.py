"""
LightGBM Training Script
Behavioral Risk Agent (per requirement: use LightGBM).
Uses: weekly_behavior + transaction aggregates from transactions.csv
"""

import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
import shap
import joblib
import yaml
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ──
random.seed(42)
np.random.seed(42)

# ── Paths ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
        return yaml.safe_load(f)

def load_thresholds():
    with open(os.path.join(ROOT, "config", "thresholds.yaml"), "r") as f:
        return yaml.safe_load(f)

def _pick_id_col(df: pd.DataFrame) -> str:
    return "user_id" if "user_id" in df.columns else "customer_id"


def _load_weekly_and_tx_features():
    weekly = pd.read_csv(os.path.join(ROOT, "data", "weekly_behavior.csv"))
    id_col = _pick_id_col(weekly)
    weekly[id_col] = weekly[id_col].astype(str)
    weekly["week_number"] = pd.to_numeric(weekly["week_number"], errors="coerce").fillna(0).astype(int)

    # Transactions aggregates computed in chunks (transactions.csv can be very large)
    tx_path = os.path.join(ROOT, "data", "transactions.csv")
    base = pd.Timestamp(f"{int(weekly['year'].mode().iloc[0])}-01-01")
    agg_chunks = []

    usecols = ["user_id", "customer_id", "date", "amount", "type", "txn_type", "category"]
    # Keep only existing columns to avoid pandas warning
    preview = pd.read_csv(tx_path, nrows=5)
    existing = [c for c in usecols if c in preview.columns]

    for chunk in pd.read_csv(tx_path, chunksize=200000, usecols=existing):
        tx_id_col = _pick_id_col(chunk)
        chunk[tx_id_col] = chunk[tx_id_col].astype(str)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date"])
        if len(chunk) == 0:
            continue

        chunk["week_number"] = (((chunk["date"] - base).dt.days // 7) + 1).clip(1, 52).astype(int)
        chunk["amount"] = pd.to_numeric(chunk.get("amount", 0), errors="coerce").fillna(0.0)

        tx_type = chunk["type"] if "type" in chunk.columns else chunk.get("txn_type", "")
        kind = tx_type.astype(str).str.upper()
        cat = chunk.get("category", "").astype(str).str.upper()

        chunk["is_debit"] = (kind == "DEBIT").astype(int)
        chunk["is_credit"] = (kind == "CREDIT").astype(int)
        chunk["is_failed"] = (kind == "FAILED").astype(int)
        chunk["is_upi_lending"] = cat.str.contains("UPI_LENDING_APP", na=False).astype(int)

        chunk["amount_debit"] = chunk["amount"] * chunk["is_debit"]
        chunk["amount_credit"] = chunk["amount"] * chunk["is_credit"]
        chunk["amount_upi_lending"] = chunk["amount"] * chunk["is_upi_lending"]

        gb = chunk.groupby([tx_id_col, "week_number"], as_index=False).agg(
            tx_debit_amount_7d=("amount_debit", "sum"),
            tx_credit_amount_7d=("amount_credit", "sum"),
            tx_debit_count_7d=("is_debit", "sum"),
            tx_failed_count_7d=("is_failed", "sum"),
            tx_upi_lending_amount_7d=("amount_upi_lending", "sum"),
        )
        gb = gb.rename(columns={tx_id_col: id_col})
        agg_chunks.append(gb)

    if agg_chunks:
        tx_agg = pd.concat(agg_chunks, ignore_index=True)
        tx_agg = tx_agg.groupby([id_col, "week_number"], as_index=False).agg(
            tx_debit_amount_7d=("tx_debit_amount_7d", "sum"),
            tx_credit_amount_7d=("tx_credit_amount_7d", "sum"),
            tx_debit_count_7d=("tx_debit_count_7d", "sum"),
            tx_failed_count_7d=("tx_failed_count_7d", "sum"),
            tx_upi_lending_amount_7d=("tx_upi_lending_amount_7d", "sum"),
        )
    else:
        tx_agg = pd.DataFrame(columns=[id_col, "week_number",
                                       "tx_debit_amount_7d", "tx_credit_amount_7d",
                                       "tx_debit_count_7d", "tx_failed_count_7d",
                                       "tx_upi_lending_amount_7d"])

    df = weekly.merge(tx_agg, on=[id_col, "week_number"], how="left")
    for c in ["tx_debit_amount_7d", "tx_credit_amount_7d", "tx_debit_count_7d", "tx_failed_count_7d", "tx_upi_lending_amount_7d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def train_lightgbm():
    print("=" * 70)
    print("  LightGBM Training — Behavioral Risk Agent")
    print("=" * 70)

    config = load_config()
    thresholds = load_thresholds()
    lgbm_cfg = config["lightgbm"]
    FEATURES = config["features"]["tabular"]
    TARGET = "will_default_next_30d"
    CLASSIFICATION_THRESHOLD = thresholds["classification_threshold"]

    # ── Load Data ──
    df = _load_weekly_and_tx_features()
    
    print(f"  Loaded {len(df)} rows, {df[TARGET].mean():.1%} positive rate")

    # ── Temporal Split: weeks 1-40 train, 41-52 test ──
    train_df = df[df["week_number"] <= 40].copy()
    test_df = df[df["week_number"] > 40].copy()
    print(f"  Train: {len(train_df)} rows (weeks 1-40)")
    print(f"  Test:  {len(test_df)} rows (weeks 41-52)")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    # ── 5-Fold Stratified CV on training data ──
    oof_preds = np.zeros(len(X_train))
    fold_aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_iterations = []
    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n  --- Fold {fold_idx + 1}/5 ---")
        X_trn, X_val = X_train[trn_idx], X_train[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]

        params = {
            "objective": lgbm_cfg["objective"],
            "metric": lgbm_cfg["metric"],
            "learning_rate": lgbm_cfg["learning_rate"],
            "num_leaves": lgbm_cfg["num_leaves"],
            "max_depth": lgbm_cfg["max_depth"],
            "min_child_samples": lgbm_cfg["min_child_samples"],
            "feature_fraction": lgbm_cfg["feature_fraction"],
            "bagging_fraction": lgbm_cfg["bagging_fraction"],
            "bagging_freq": lgbm_cfg["bagging_freq"],
            "lambda_l1": lgbm_cfg["lambda_l1"],
            "lambda_l2": lgbm_cfg["lambda_l2"],
            "is_unbalance": lgbm_cfg["is_unbalance"],
            "random_state": lgbm_cfg["random_state"],
            "n_jobs": lgbm_cfg["n_jobs"],
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X_trn, label=y_trn, feature_name=FEATURES)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=lgbm_cfg["n_estimators"],
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        best_iterations.append(model.best_iteration)
        print(f"  Fold {fold_idx + 1} AUC: {fold_auc:.4f} (best iter: {model.best_iteration})")

    print(f"\n  CV Mean AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

    # ── Retrain on ALL training data ──
    best_n = int(np.mean(best_iterations))
    print(f"\n  Retraining on full train set with n_estimators={best_n}...")
    dtrain_full = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    final_model = lgb.train(
        params, dtrain_full,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(period=200)],
    )

    # ── Evaluate on test set ──
    test_preds = final_model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_preds)
    test_prauc = average_precision_score(y_test, test_preds)
    test_binary = (test_preds >= CLASSIFICATION_THRESHOLD).astype(int)

    print(f"\n  Test ROC-AUC: {test_auc:.4f}")
    print(f"  Test PR-AUC:  {test_prauc:.4f}")
    print(f"\n  Classification Report (threshold={CLASSIFICATION_THRESHOLD}):")
    print(classification_report(y_test, test_binary, target_names=["No Default", "Default"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, test_binary))

    # ── SHAP Values ──
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    # Use a sample for SHAP plots to avoid memory issues
    sample_size = min(5000, len(X_test))
    X_test_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(X_test_sample)

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)

    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=FEATURES,
                      show=False, plot_size=(12, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "reports", "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved reports/shap_summary.png")

    # SHAP Beeswarm Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=FEATURES,
                      plot_type="bar", show=False, plot_size=(12, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "reports", "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved reports/shap_beeswarm.png")

    # ── Save artifacts ──
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    model_path = os.path.join(ROOT, "models", "lgbm_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"  Saved model: {model_path}")

    oof_path = os.path.join(ROOT, "models", "lgbm_oof_preds.npy")
    np.save(oof_path, oof_preds)
    print(f"  Saved OOF predictions: {oof_path}")

    print(f"\n  [OK] LightGBM training complete!")
    return final_model, test_auc


if __name__ == "__main__":
    train_lightgbm()
