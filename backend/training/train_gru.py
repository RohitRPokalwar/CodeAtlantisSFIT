"""
GRU Training Script
Sequence-based risk prediction using 8-week temporal windows.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import yaml
import os
import sys

# ── Reproducibility ──
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
        return yaml.safe_load(f)


class GRUModel(nn.Module):
    """GRU-based temporal risk predictor."""

    def __init__(self, input_size=12, hidden1=64, hidden2=32, dropout=0.3):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden1,
                           batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden1, hidden_size=hidden2,
                           batch_first=True)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features) = (B, 8, 12)
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = out[:, -1, :]  # Take ONLY the LAST timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sequences(df, features, target, seq_len=8):
    """Build temporal sequences with correct label alignment.
    Label at position i comes from week (i + seq_len), NOT week i.
    """
    sequences, labels, customer_ids, week_numbers = [], [], [], []
    for cid, group in df.sort_values("week_number").groupby("customer_id"):
        feat_array = group[features].values
        label_array = group[target].values
        weeks = group["week_number"].values
        for i in range(seq_len, len(feat_array)):
            # Input: weeks [i-seq_len : i]
            # Label: label at week i (FUTURE label)
            sequences.append(feat_array[i - seq_len: i])
            labels.append(label_array[i])
            customer_ids.append(cid)
            week_numbers.append(weeks[i])
    return (np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            customer_ids, week_numbers)


def train_gru():
    print("=" * 70)
    print("  GRU Training - Temporal Sequence Risk Model")
    print("=" * 70)

    config = load_config()
    gru_cfg = config["gru"]
    FEATURES = config["features"]["sequence"]
    TARGET = "will_default_next_30d"
    SEQ_LEN = gru_cfg["seq_len"]

    # ── Load Data ──
    df = pd.read_csv(os.path.join(ROOT, "data", "weekly_behavioral_features.csv"))
    
    # Explicit Segment Mapping (Bank-Grade Consistency)
    SEGMENT_MAP = {"salaried": 0, "self-employed": 1, "farmer": 2, "freelancer": 3, "student": 4, "other": 5}
    if "customer_segment" in df.columns:
        df["customer_segment"] = df["customer_segment"].map(SEGMENT_MAP).fillna(5).astype(int)
    
    print(f"  Loaded {len(df)} rows")

    # ── Temporal Split ──
    train_df = df[df["week_number"] <= 40].copy()
    test_df = df[df["week_number"] > 40].copy()

    # ── Build Sequences ──
    print(f"  Building sequences (seq_len={SEQ_LEN})...")
    train_seqs, train_labels, train_cids, train_weeks = build_sequences(
        train_df, FEATURES, TARGET, SEQ_LEN)
    test_seqs, test_labels, test_cids, test_weeks = build_sequences(
        test_df, FEATURES, TARGET, SEQ_LEN)

    print(f"  Train sequences: {train_seqs.shape}")
    print(f"  Test sequences:  {test_seqs.shape}")

    # ── Scale features (fit on train ONLY) ──
    n_train, seq_l, n_feat = train_seqs.shape
    scaler = StandardScaler()
    train_seqs_2d = train_seqs.reshape(-1, n_feat)
    scaler.fit(train_seqs_2d)
    train_seqs = scaler.transform(train_seqs_2d).reshape(n_train, seq_l, n_feat)

    n_test = test_seqs.shape[0]
    test_seqs_2d = test_seqs.reshape(-1, n_feat)
    test_seqs = scaler.transform(test_seqs_2d).reshape(n_test, seq_l, n_feat)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join(ROOT, "models", "gru_scaler.pkl"))
    print("  Saved scaler: models/gru_scaler.pkl")

    # ── Train/Val split within training data (stratified) ──
    from sklearn.model_selection import train_test_split
    X_trn, X_val, y_trn, y_val = train_test_split(
        train_seqs, train_labels, test_size=0.2,
        stratify=train_labels, random_state=42)

    # ── DataLoaders ──
    batch_size = gru_cfg["batch_size"]
    train_ds = SequenceDataset(X_trn, y_trn)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Model ──
    model = GRUModel(
        input_size=len(FEATURES),
        hidden1=gru_cfg["hidden1"],
        hidden2=gru_cfg["hidden2"],
        dropout=gru_cfg["dropout"]
    ).to(DEVICE)
    print(f"\n  Model on device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.BCELoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=gru_cfg["lr"],
                                 weight_decay=gru_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    # ── Training Loop ──
    best_val_auc = 0
    patience_counter = 0
    PATIENCE = 10
    pw = gru_cfg["pos_weight"]

    for epoch in range(gru_cfg["epochs"]):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            # Manual pos_weight: weight = 1 for neg, pos_weight for pos
            weights = torch.where(y_batch == 1, torch.tensor(pw).to(DEVICE),
                                  torch.tensor(1.0).to(DEVICE))
            loss = (criterion(preds, y_batch) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ──
        model.eval()
        val_preds_list, val_labels_list = [], []
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                weights = torch.where(y_batch == 1, torch.tensor(pw).to(DEVICE),
                                      torch.tensor(1.0).to(DEVICE))
                loss = (criterion(preds, y_batch) * weights).mean()
                val_losses.append(loss.item())
                val_preds_list.append(preds.cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())

        val_preds_np = np.concatenate(val_preds_list)
        val_labels_np = np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels_np, val_preds_np)
        scheduler.step(np.mean(val_losses))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{gru_cfg['epochs']} | "
                  f"Train Loss: {np.mean(train_losses):.4f} | "
                  f"Val Loss: {np.mean(val_losses):.4f} | "
                  f"Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(ROOT, "models", "gru_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Load best model & evaluate on test ──
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "gru_model.pt"),
                                      weights_only=True))
    model.eval()
    test_preds_list, test_labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            test_preds_list.append(preds.cpu().numpy())
            test_labels_list.append(y_batch.numpy())

    test_preds_np = np.concatenate(test_preds_list)
    test_labels_np = np.concatenate(test_labels_list)
    test_auc = roc_auc_score(test_labels_np, test_preds_np)
    test_binary = (test_preds_np >= 0.45).astype(int)

    print(f"\n  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(test_labels_np, test_binary,
                                target_names=["No Default", "Default"]))

    # ── Save OOF predictions ──
    # Generate predictions for the train set as OOF proxy
    model.eval()
    train_ds_full = SequenceDataset(train_seqs, train_labels)
    train_loader_full = DataLoader(train_ds_full, batch_size=batch_size, shuffle=False)
    oof_list = []
    with torch.no_grad():
        for X_batch, _ in train_loader_full:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            oof_list.append(preds.cpu().numpy())
    gru_oof = np.concatenate(oof_list)
    np.save(os.path.join(ROOT, "models", "gru_oof_preds.npy"), gru_oof)
    print(f"  Saved OOF predictions: models/gru_oof_preds.npy ({len(gru_oof)} samples)")

    # Save metadata for ensemble alignment
    np.save(os.path.join(ROOT, "models", "gru_train_cids.npy"), np.array(train_cids[:len(train_seqs)]))
    np.save(os.path.join(ROOT, "models", "gru_train_weeks.npy"), np.array(train_weeks[:len(train_seqs)]))

    print(f"\n  [OK] GRU training complete!")
    return model, test_auc


if __name__ == "__main__":
    train_gru()
