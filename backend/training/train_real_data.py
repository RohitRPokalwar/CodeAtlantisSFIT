"""
Praeventix — Train All Models on Real Lending Club Dataset
===========================================================
Uses the archive dataset (df_2014-18_selected.csv) with 2M+ real loan records.
Target: loan_status_binary (0 = default/charged-off, 1 = fully paid)

Models trained:
  1. LightGBM       — Tabular risk scoring (5-fold CV, SHAP)
  2. GRU (PyTorch)  — Temporal sequence model (sliding windows)
  3. Ensemble       — Meta-learner combining LightGBM + GRU
  4. Isolation Forest — Anomaly detection
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARCHIVE_PATH = os.path.join(os.path.dirname(ROOT), "archive (4)", "df_2014-18_selected.csv")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "total_rec_late_fee", "recoveries", "last_pymnt_amnt",
    "loan_amnt_div_instlmnt", "debt_settlement_flag", "loan_age",
    "total_rec_int", "out_prncp", "time_since_last_credit_pull",
    "time_since_last_payment", "int_rate%", "total_rec_prncp"
]
TARGET = "loan_status_binary"


def load_data(sample_frac=0.25):
    """Load and prepare the combined dataset (Lending Club + Synthetic)."""
    print("\n" + "=" * 60)
    print("  PRAEVENTIX — Combined Data Model Training Pipeline")
    print("=" * 60)

    print(f"\n📂 Loading REAL dataset from: {ARCHIVE_PATH}")
    df_real = pd.read_csv(ARCHIVE_PATH)
    
    SYNTHETIC_PATH = os.path.join(ROOT, "data", "synthetic_archive.csv")
    print(f"📂 Loading SYNTHETIC dataset from: {SYNTHETIC_PATH}")
    if os.path.exists(SYNTHETIC_PATH):
        df_synthetic = pd.read_csv(SYNTHETIC_PATH)
    else:
        df_synthetic = pd.DataFrame()
        print("   ⚠ Synthetic dataset missing, check backend/data!")

    # Ensure synthetic dataset is compatible (subsetting to FEATURE_COLS + target)
    if not df_synthetic.empty:
        # Flip synthetic target mapping: original Lending Club uses 0 for default 
        if "loan_status_binary" in df_synthetic.columns:
            df_synthetic[TARGET] = df_synthetic["loan_status_binary"]
            
    print(f"   Real dataset: {len(df_real):,} rows")
    print(f"   Synthetic dataset: {len(df_synthetic):,} rows")

    # Combine datasets
    df = pd.concat([df_real, df_synthetic], ignore_index=True)
    print(f"   Combined dataset: {len(df):,} rows")

    # For hackathon speed, sample the data
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"   Sampled {sample_frac*100:.0f}%: {len(df):,} rows")

    # Flip target: 0 = good, 1 = default (for risk scoring)
    df["target"] = 1 - df[TARGET]
    print(f"\n📊 Target distribution (risk = default):")
    print(f"   Low risk (0):  {(df['target'] == 0).sum():,} ({(df['target'] == 0).mean()*100:.1f}%)")
    print(f"   High risk (1): {(df['target'] == 1).sum():,} ({(df['target'] == 1).mean()*100:.1f}%)")

    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y, df


# ─────────────────────────────────────────────────
# MODEL 1: LightGBM with 5-Fold CV + SHAP
# ─────────────────────────────────────────────────

def train_lightgbm(X, y):
    """Train LightGBM with 5-fold stratified CV and SHAP."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, classification_report

    print("\n" + "─" * 60)
    print("  🌳 MODEL 1: LightGBM (Gradient Boosting)")
    print("─" * 60)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "is_unbalance": True,
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    fold_aucs = []
    best_model = None
    best_auc = 0

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(auc)
        print(f"   Fold {fold}: AUC = {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    elapsed = time.time() - t0
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n   ✅ Overall OOF AUC: {overall_auc:.4f}")
    print(f"   ⏱  Training time: {elapsed:.1f}s")

    # Save model and OOF predictions
    joblib.dump(best_model, os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    np.save(os.path.join(MODELS_DIR, "lgbm_oof_preds.npy"), oof_preds)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, best_model.feature_importances_))
    print(f"\n   📊 Top Features:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:6]:
        print(f"      {feat}: {imp}")

    # SHAP
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"\n   🔍 Computing SHAP explanations...")
        explainer = shap.TreeExplainer(best_model)
        shap_sample = X[:min(2000, len(X))]
        shap_values = explainer.shap_values(shap_sample)

        # Handle binary output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, shap_sample,
                          feature_names=FEATURE_COLS, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, shap_sample,
                          feature_names=FEATURE_COLS, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_beeswarm.png"), dpi=150)
        plt.close()
        print(f"   📈 SHAP plots saved to reports/")
    except Exception as e:
        print(f"   ⚠ SHAP plot skipped: {e}")

    # Save training indices for ensemble
    np.save(os.path.join(MODELS_DIR, "lgbm_train_indices.npy"), np.arange(len(X)))

    return best_model, oof_preds


# ─────────────────────────────────────────────────
# MODEL 2: GRU (Temporal Sequence Model)
# ─────────────────────────────────────────────────

def train_gru(X, y):
    """Train GRU for temporal pattern detection.
    Simulates 8-step sequences from feature rows using sliding windows."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("\n" + "─" * 60)
    print("  🧠 MODEL 2: GRU (Temporal Pattern Detection)")
    print("─" * 60)

    SEQ_LEN = 8
    HIDDEN1 = 64
    HIDDEN2 = 32
    DROPOUT = 0.3
    EPOCHS = 30
    BATCH_SIZE = 512
    LR = 0.001
    POS_WEIGHT = 4.0

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "gru_scaler.pkl"))

    # Create sequences: use sliding windows with small perturbations
    # This simulates temporal evolution for non-sequential data
    n_samples = len(X_scaled) - SEQ_LEN + 1
    n_features = X_scaled.shape[1]
    sequences = np.zeros((n_samples, SEQ_LEN, n_features))
    seq_targets = np.zeros(n_samples)

    for i in range(n_samples):
        sequences[i] = X_scaled[i:i + SEQ_LEN]
        seq_targets[i] = y[i + SEQ_LEN - 1]

    print(f"   Sequences: {sequences.shape[0]:,} × {SEQ_LEN} steps × {n_features} features")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, seq_targets, test_size=0.2, random_state=42, stratify=seq_targets
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # GRU Model
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden1, hidden2, dropout):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden1, batch_first=True, dropout=dropout, num_layers=2)
            self.fc1 = nn.Linear(hidden1, hidden2)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden2, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            _, h = self.gru(x)
            out = self.relu(self.fc1(h[-1]))
            out = self.dropout(out)
            return self.fc2(out).squeeze(-1)

    model = GRUModel(n_features, HIDDEN1, HIDDEN2, DROPOUT).to(device)
    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Data loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val).to(device),
        torch.FloatTensor(y_val).to(device)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Training loop
    t0 = time.time()
    best_val_auc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = torch.sigmoid(model(xb))
                val_preds.extend(out.cpu().numpy())
                val_true.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(val_true, val_preds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:2d}/{EPOCHS}: loss={train_loss/len(train_loader):.4f}  val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "gru_model.pt"))

    elapsed = time.time() - t0
    print(f"\n   ✅ Best Validation AUC: {best_val_auc:.4f}")
    print(f"   ⏱  Training time: {elapsed:.1f}s")

    # Generate OOF predictions for ensemble
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "gru_model.pt"), weights_only=True))
    model.eval()
    all_preds = []
    all_ds = TensorDataset(torch.FloatTensor(sequences).to(device))
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for (xb,) in all_loader:
            out = torch.sigmoid(model(xb))
            all_preds.extend(out.cpu().numpy())

    gru_oof = np.array(all_preds)
    np.save(os.path.join(MODELS_DIR, "gru_oof_preds.npy"), gru_oof)

    # Save metadata for inference
    np.save(os.path.join(MODELS_DIR, "gru_train_cids.npy"), np.arange(n_samples))
    np.save(os.path.join(MODELS_DIR, "gru_train_weeks.npy"), np.arange(n_samples))

    return model, gru_oof, n_samples


# ─────────────────────────────────────────────────
# MODEL 3: Ensemble Meta-Learner
# ─────────────────────────────────────────────────

def train_ensemble(lgbm_oof, gru_oof, y, n_gru_samples):
    """Train a meta-learner combining LightGBM + GRU OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n" + "─" * 60)
    print("  🔗 MODEL 3: Ensemble Meta-Learner")
    print("─" * 60)

    # Align predictions (GRU may have fewer samples due to sequence windowing)
    n = min(len(lgbm_oof), len(gru_oof))
    offset = len(lgbm_oof) - n
    lgbm_aligned = lgbm_oof[offset:offset + n]
    gru_aligned = gru_oof[:n]
    y_aligned = y[offset:offset + n]

    # Stack as meta-features
    meta_X = np.column_stack([lgbm_aligned, gru_aligned])
    print(f"   Meta-features shape: {meta_X.shape}")

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X, y_aligned)

    ensemble_preds = meta_model.predict_proba(meta_X)[:, 1]
    auc = roc_auc_score(y_aligned, ensemble_preds)
    print(f"\n   ✅ Ensemble AUC: {auc:.4f}")

    # Show model weights
    coefs = meta_model.coef_[0]
    print(f"   📊 Model weights: LightGBM={coefs[0]:.3f}, GRU={coefs[1]:.3f}")

    joblib.dump(meta_model, os.path.join(MODELS_DIR, "ensemble_meta.pkl"))
    return meta_model


# ─────────────────────────────────────────────────
# MODEL 4: Isolation Forest (Anomaly Detection)
# ─────────────────────────────────────────────────

def train_isolation_forest(X):
    """Train Isolation Forest for anomaly detection."""
    from sklearn.ensemble import IsolationForest

    print("\n" + "─" * 60)
    print("  🔎 MODEL 4: Isolation Forest (Anomaly Detection)")
    print("─" * 60)

    t0 = time.time()
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    iso.fit(X)
    elapsed = time.time() - t0

    anomalies = (iso.predict(X) == -1).sum()
    print(f"   Anomalies detected: {anomalies:,} / {len(X):,} ({anomalies/len(X)*100:.1f}%)")
    print(f"   ⏱  Training time: {elapsed:.1f}s")

    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    print(f"\n   ✅ Isolation Forest saved")
    return iso


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    total_start = time.time()

    # Load data (25% sample for hackathon speed)
    X, y, df = load_data(sample_frac=0.25)

    # Train all models
    lgbm_model, lgbm_oof = train_lightgbm(X, y)
    gru_model, gru_oof, n_gru = train_gru(X, y)
    ensemble = train_ensemble(lgbm_oof, gru_oof, y, n_gru)
    iso_forest = train_isolation_forest(X)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  🎉 ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n   Total training time: {total_time:.1f}s")
    print(f"\n   📁 Model artifacts saved to: {MODELS_DIR}")
    print(f"   📈 SHAP reports saved to: {REPORTS_DIR}")
    print()

    # List saved artifacts
    for f in sorted(os.listdir(MODELS_DIR)):
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        print(f"   ✅ {f} ({size/1024:.0f} KB)")
    print()
