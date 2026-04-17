"""
LSTM Training Script (Time-based Trend Model)
Uses: weekly_behavior
Target: will_default_next_30d (per user, week-level sequences)

Updates:
  - Saves trained model + scaler + metadata to models/lstm_model.pkl
  - Mirrors save structure of xgb_model.pkl for consistent backend loading
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEVICE = torch.device("cpu")


# ── Config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _pick_id_col(df: pd.DataFrame) -> str:
    if "user_id" in df.columns:
        return "user_id"
    if "customer_id" in df.columns:
        return "customer_id"
    raise ValueError(f"No user_id / customer_id column found. Columns: {df.columns.tolist()}")


# ── Dataset & Model ──────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        return self.fc(last).squeeze(-1)


# ── Sequence Building ────────────────────────────────────────────────────────

def build_last_window_per_user(
    df: pd.DataFrame,
    features: list,
    target: str,
    seq_len: int,
    label_week: int = 52,
) -> tuple:
    id_col = _pick_id_col(df)

    df["week_number"] = (
        pd.to_numeric(df["week_number"], errors="coerce").fillna(0).astype(int)
    )
    df[target] = (
        pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    )

    start_week = max(1, label_week - seq_len + 1)
    df = df[(df["week_number"] >= start_week) & (df["week_number"] <= label_week)]
    df = df.sort_values([id_col, "week_number"])

    X_list, y_list = [], []

    for uid, g in df.groupby(id_col):
        label_rows = g[g["week_number"] == label_week]
        if label_rows.empty:
            continue

        y    = int(label_rows.iloc[-1][target])
        vals = g[features].to_numpy(dtype=np.float32)

        if len(vals) >= seq_len:
            seq = vals[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(vals), len(features)), dtype=np.float32)
            seq = np.vstack([pad, vals])

        X_list.append(seq)
        y_list.append(y)

    return np.array(X_list), np.array(y_list)


# ── Main Training ─────────────────────────────────────────────────────────────

def train_lstm():
    print("=" * 70)
    print("  LSTM Training — Time-based Trend Model  (15,000 Users)")
    print("=" * 70)

    cfg      = load_config()
    FEATURES = cfg["features"]["sequence"]
    lstm_cfg = cfg["lstm"]

    # ── Load data ──────────────────────────────────────────────────────────
    path = os.path.join(ROOT, "data", "weekly_behavior.csv")
    df   = pd.read_csv(path)
    print(f"\n  weekly_behavior loaded: {df.shape}")

    X_all, y_all = build_last_window_per_user(
        df, FEATURES, "will_default_next_30d", lstm_cfg["seq_len"]
    )

    n_total   = len(y_all)
    n_default = int(y_all.sum())
    n_normal  = n_total - n_default

    print(f"\n  Dataset summary:")
    print(f"    Sequences shape : {X_all.shape}  (users × weeks × features)")
    print(f"    Total users     : {n_total:,}")
    print(f"    Defaults (1)    : {n_default:,}  ({n_default/n_total*100:.2f}%)")
    print(f"    Non-default (0) : {n_normal:,}  ({n_normal/n_total*100:.2f}%)")

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\n  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── Scale (per-feature across time steps) ─────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    X_test  = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    # ── Loaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=64)

    # ── Model ──────────────────────────────────────────────────────────────
    input_size  = X_train.shape[2]
    hidden_size = lstm_cfg.get("hidden_size", 64)
    num_layers  = lstm_cfg.get("num_layers",  2)
    dropout     = lstm_cfg.get("dropout",     0.3)
    n_epochs    = lstm_cfg.get("epochs",      10)

    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(DEVICE)

    # Dynamic pos_weight for class imbalance
    pos        = np.sum(y_train == 1)
    neg        = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg / (pos + 1e-8)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Training for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Epoch [{epoch:02d}/{n_epochs}]  loss={epoch_loss/len(train_loader):.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            prob = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            all_probs.append(prob)
            all_labels.append(yb.numpy())

    tp = np.concatenate(all_probs)
    tl = np.concatenate(all_labels)

    auc = roc_auc_score(tl, tp)
    pr  = average_precision_score(tl, tp)

    # Optimal threshold via F1
    precision, recall, thresholds = precision_recall_curve(tl, tp)
    f1          = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx    = np.argmax(f1[:-1])          # thresholds has len-1 vs precision/recall
    best_thresh = float(thresholds[best_idx])
    preds       = (tp >= best_thresh).astype(int)

    print(f"\n  Test ROC-AUC : {auc:.4f}")
    print(f"  Test PR-AUC  : {pr:.4f}")
    print(f"\n  Best Threshold (max F1): {best_thresh:.4f}")
    print(f"  Precision @ best : {precision[best_idx]:.4f}")
    print(f"  Recall    @ best : {recall[best_idx]:.4f}")
    print(f"\n  Classification Report (threshold={best_thresh:.3f}):")
    print(classification_report(tl, preds, target_names=["No Default", "Default"]))

    # ── Save to models/lstm_model.pkl ─────────────────────────────────────
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "lstm_model.pkl")

    joblib.dump(
        {
            # Core artifacts needed for inference
            "model_state_dict" : model.state_dict(),       # weights
            "model_config"     : {                          # architecture params
                "input_size"  : input_size,
                "hidden_size" : hidden_size,
                "num_layers"  : num_layers,
                "dropout"     : dropout,
            },
            "scaler"           : scaler,                    # StandardScaler for inference
            "features"         : FEATURES,                  # feature list (order matters)
            "seq_len"          : lstm_cfg["seq_len"],       # required by inference pipeline
            "best_threshold"   : best_thresh,               # tuned decision threshold

            # Metadata (mirrors xgb_model.pkl)
            "n_users_trained"  : n_total,
            "default_rate"     : round(n_default / n_total, 6),
            "pos_weight"       : round(float(pos_weight.item()), 4),
            "test_roc_auc"     : round(auc, 6),
            "test_pr_auc"      : round(pr,  6),
        },
        out_path,
    )

    print(f"\n  Model saved → {out_path}")
    print("=" * 70)

    return model


# ── Inference helper (used by backend) ───────────────────────────────────────

def load_lstm_for_inference(models_dir: str = None) -> dict:
    """
    Load lstm_model.pkl and reconstruct the LSTMModel ready for inference.

    Usage in backend:
        bundle = load_lstm_for_inference()
        model  = bundle["model"]      # nn.Module in eval mode
        scaler = bundle["scaler"]     # StandardScaler
        feats  = bundle["features"]
        thresh = bundle["best_threshold"]
    """
    if models_dir is None:
        models_dir = os.path.join(ROOT, "models")

    bundle = joblib.load(os.path.join(models_dir, "lstm_model.pkl"))

    cfg   = bundle["model_config"]
    model = LSTMModel(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    bundle["model"] = model          # replace state_dict with live nn.Module
    return bundle


if __name__ == "__main__":
    train_lstm()