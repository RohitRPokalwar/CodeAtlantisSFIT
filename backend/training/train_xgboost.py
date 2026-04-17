"""
XGBoost Training Script (Financial Strength) — Updated for 15,000 Users Dataset
Uses: customers + salary + payments
Target: will_default_next_30d (aggregated per user from weekly_behavior)

Fixes applied:
  - Replaced incorrect DataFrame.get() calls with proper column-existence checks
  - Added scale_pos_weight for class imbalance handling
  - Added train-set evaluation to detect overfitting
  - Added dataset shape logging after every merge (detects silent data loss)
  - Added early_stopping_rounds with eval_set
  - Saved feature names + model metadata alongside pickle
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from xgboost import XGBClassifier

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(ROOT, "config", "model_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _pick_id_col(df: pd.DataFrame) -> str:
    """Return the correct user identifier column name."""
    if "user_id" in df.columns:
        return "user_id"
    if "customer_id" in df.columns:
        return "customer_id"
    raise ValueError(f"No user_id / customer_id column found. Columns: {df.columns.tolist()}")


def _log_shape(df: pd.DataFrame, label: str) -> None:
    print(f"    [{label}] shape={df.shape}  |  "
          f"nulls={df.isnull().sum().sum()}  |  "
          f"unique_users={df[_pick_id_col(df)].nunique() if any(c in df.columns for c in ['user_id','customer_id']) else 'N/A'}")


# ── Feature Engineering ───────────────────────────────────────────────────────

def _aggregate_salary(salary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate salary table to one row per user."""
    id_col = _pick_id_col(salary_df)
    salary_df = salary_df.copy()

    salary_df["salary_amount"] = pd.to_numeric(
        salary_df["salary_amount"], errors="coerce"
    ).fillna(0.0)

    salary_df["credit_delay_days"] = pd.to_numeric(
        salary_df["credit_delay_days"], errors="coerce"
    ).fillna(0.0)

    # FIX: use column existence check instead of DataFrame.get()
    if "bonus_amount" in salary_df.columns:
        salary_df["bonus_amount"] = pd.to_numeric(
            salary_df["bonus_amount"], errors="coerce"
        ).fillna(0.0)
    else:
        salary_df["bonus_amount"] = 0.0

    g = salary_df.groupby(id_col, as_index=False).agg(
        salary_mean=("salary_amount", "mean"),
        salary_std=("salary_amount", "std"),
        salary_max_delay_days=("credit_delay_days", "max"),
        salary_bonus_sum=("bonus_amount", "sum"),
        salary_count=("salary_amount", "count"),          # NEW: number of salary records
    )
    g["salary_std"] = g["salary_std"].fillna(0.0)
    return g


def _aggregate_payments(pay_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payments table to one row per user."""
    id_col = _pick_id_col(pay_df)
    pay_df = pay_df.copy()

    numeric_cols = [
        "emi_amount", "emi_paid", "days_late",
        "outstanding_balance", "penalty_applied",
    ]
    for col in numeric_cols:
        if col in pay_df.columns:
            pay_df[col] = pd.to_numeric(pay_df[col], errors="coerce").fillna(0.0)

    # FIX: use column existence check instead of DataFrame.get()
    days_late_col   = pay_df["days_late"]          if "days_late"          in pay_df.columns else pd.Series(0, index=pay_df.index)
    emi_paid_col    = pay_df["emi_paid"]            if "emi_paid"           in pay_df.columns else pd.Series(0, index=pay_df.index)
    penalty_col     = pay_df["penalty_applied"]     if "penalty_applied"    in pay_df.columns else pd.Series(0, index=pay_df.index)
    outstanding_col = pay_df["outstanding_balance"] if "outstanding_balance" in pay_df.columns else pd.Series(0, index=pay_df.index)

    pay_df["is_late"]    = (days_late_col   > 0).astype(int)
    pay_df["is_missed"]  = (emi_paid_col   <= 0).astype(int)
    pay_df["has_penalty"] = (penalty_col    > 0).astype(int)

    g = pay_df.groupby(id_col, as_index=False).agg(
        pay_late_rate=("is_late",            "mean"),
        pay_mean_days_late=("days_late",     "mean") if "days_late" in pay_df.columns else ("is_late", "mean"),
        pay_missed_rate=("is_missed",        "mean"),
        pay_penalty_rate=("has_penalty",     "mean"),
        pay_outstanding_ratio=("outstanding_balance", "mean") if "outstanding_balance" in pay_df.columns else ("is_late", "mean"),
        pay_count=("is_late",                "count"),   # NEW: total EMI records
    )
    return g


def _build_target(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 if user ever flagged will_default_next_30d in weeks 41+ (later half of year).
    """
    id_col = _pick_id_col(weekly_df)
    weekly_df = weekly_df.copy()
    weekly_df["will_default_next_30d"] = (
        pd.to_numeric(weekly_df["will_default_next_30d"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    g = (
        weekly_df[weekly_df["week_number"] >= 41]
        .groupby(id_col, as_index=False)
        .agg(will_default_next_30d=("will_default_next_30d", "max"))
    )
    return g


# ── Main Training ─────────────────────────────────────────────────────────────

def train_xgboost():
    print("=" * 70)
    print("  XGBoost Training — Financial Strength Model  (15,000 Users)")
    print("=" * 70)

    cfg      = load_config()
    xgb_cfg  = cfg["xgboost"]
    FEATURES = cfg["features"]["financial"]

    # ── Load raw data ──────────────────────────────────────────────────────
    data_dir  = os.path.join(ROOT, "data")
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    salary    = pd.read_csv(os.path.join(data_dir, "salary.csv"))
    payments  = pd.read_csv(os.path.join(data_dir, "payments.csv"))
    weekly    = pd.read_csv(os.path.join(data_dir, "weekly_behavior.csv"))

    print("\n  Raw data loaded:")
    print(f"    customers      : {customers.shape}")
    print(f"    salary         : {salary.shape}")
    print(f"    payments       : {payments.shape}")
    print(f"    weekly_behavior: {weekly.shape}")

    id_col = _pick_id_col(customers)

    # ── Aggregate ──────────────────────────────────────────────────────────
    print("\n  Aggregating features...")
    sal_agg = _aggregate_salary(salary)
    pay_agg = _aggregate_payments(payments)
    y_df    = _build_target(weekly)

    print(f"    sal_agg  : {sal_agg.shape}")
    print(f"    pay_agg  : {pay_agg.shape}")
    print(f"    target   : {y_df.shape}  |  default_rate={y_df['will_default_next_30d'].mean():.4f}")

    # ── Merge ──────────────────────────────────────────────────────────────
    print("\n  Merging datasets...")
    df = customers.copy()
    df = df.merge(sal_agg, on=id_col, how="left");  _log_shape(df, "after salary merge")
    df = df.merge(pay_agg, on=id_col, how="left");  _log_shape(df, "after payments merge")
    df = df.merge(y_df,    on=id_col, how="left");  _log_shape(df, "after target merge")

    df["will_default_next_30d"] = df["will_default_next_30d"].fillna(0).astype(int)

    # ── Build X, y ────────────────────────────────────────────────────────
    for col in FEATURES:
        if col not in df.columns:
            print(f"    [WARN] Feature '{col}' missing — filling with 0.0")
            df[col] = 0.0

    X = df[FEATURES].astype(float).values
    y = df["will_default_next_30d"].values

    n_total    = len(y)
    n_default  = int(y.sum())
    n_normal   = n_total - n_default
    class_ratio = n_normal / n_default if n_default > 0 else 1.0

    print(f"\n  Dataset summary:")
    print(f"    Total users    : {n_total:,}")
    print(f"    Defaults (1)   : {n_default:,}  ({n_default/n_total*100:.2f}%)")
    print(f"    Non-default (0): {n_normal:,}  ({n_normal/n_total*100:.2f}%)")
    print(f"    scale_pos_weight (auto): {class_ratio:.2f}")

    # ── Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = XGBClassifier(
        objective        = xgb_cfg["objective"],
        eval_metric      = xgb_cfg["eval_metric"],
        max_depth        = int(xgb_cfg["max_depth"]),
        learning_rate    = float(xgb_cfg["learning_rate"]),
        subsample        = float(xgb_cfg["subsample"]),
        colsample_bytree = float(xgb_cfg["colsample_bytree"]),
        n_estimators     = int(xgb_cfg["n_estimators"]),
        reg_lambda       = float(xgb_cfg["reg_lambda"]),
        random_state     = int(xgb_cfg["random_state"]),
        n_jobs           = int(xgb_cfg["n_jobs"]),
        scale_pos_weight = class_ratio,   # FIX: handle class imbalance automatically
        tree_method      = "hist",
        early_stopping_rounds = 20,       # NEW: prevent overfitting
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set        = eval_set,
        verbose         = False,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n  Evaluation:")
    for split_name, X_eval, y_eval in [
        ("Train", X_train, y_train),
        ("Test ", X_test,  y_test),
    ]:
        probs = model.predict_proba(X_eval)[:, 1]
        auc   = roc_auc_score(y_eval, probs)          if len(np.unique(y_eval)) > 1 else 0.5
        pr    = average_precision_score(y_eval, probs) if len(np.unique(y_eval)) > 1 else 0.0
        preds = (probs >= 0.5).astype(int)
        print(f"\n  [{split_name}]  ROC-AUC: {auc:.4f}  |  PR-AUC: {pr:.4f}")
        print(classification_report(y_eval, preds, target_names=["No Default", "Default"]))

    # ── Save model ────────────────────────────────────────────────────────
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "xgb_model.pkl")

    joblib.dump(
        {
            "model"           : model,
            "features"        : FEATURES,
            "n_users_trained" : n_total,
            "default_rate"    : round(n_default / n_total, 6),
            "scale_pos_weight": round(class_ratio, 4),
            "best_iteration"  : model.best_iteration,
        },
        out_path,
    )
    print(f"\n  Model saved → {out_path}")
    print(f"  Best iteration (early stopping): {model.best_iteration}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    train_xgboost()