"""
Generate Pre-Scored Customers using Final Architecture models
=============================================================
Writes `data/scored_customers.json` for the dashboard.

Keeps response schema backward-compatible for the existing frontend:
- `lgbm_prob`
- `gru_prob`  (now LSTM prob)
- `ensemble_prob` (blended score)
- `anomaly_flag` (rule-based)
- `risk_level`
- `shap_top3`, `all_shap`, `human_explanation`, `confidence`
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "scored_customers.json")


def _pick_id_col(df: pd.DataFrame) -> str:
    return "user_id" if "user_id" in df.columns else "customer_id"

def main():
    print("\n" + "=" * 60)
    print("  🚀 PRAEVENTIX SCORING ENGINE (FINAL ARCHITECTURE)")
    print("=" * 60)

    from inference.predict import RiskPredictor

    customers_df = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    weekly_df = pd.read_csv(os.path.join(DATA_DIR, "weekly_behavior.csv"))
    id_col = _pick_id_col(customers_df)

    predictor = RiskPredictor()
    latest_week = int(pd.to_numeric(weekly_df["week_number"], errors="coerce").fillna(0).max())
    latest_week = latest_week if latest_week > 0 else 52

    scored = []
    ids = customers_df[id_col].astype(str).tolist()
    for i, cid in enumerate(ids):
        if (i + 1) % 500 == 0:
            print(f"  Scoring {i+1}/{len(ids)}...")

        pred = predictor.predict_single(cid, latest_week)
        if pred.get("error"):
            continue

        prof = pred.get("customer_profile", {}) or {}
        record = {
            "customer_id": str(cid),
            "name": str(prof.get("name", "")),
            "city": str(prof.get("city", "Unknown")),
            "occupation": str(prof.get("occupation", "")),
            "product_type": str(prof.get("product_type", "")),
            "age": int(prof.get("age", 0) or 0),
            "monthly_salary": float(prof.get("monthly_salary", 0) or 0),
            "credit_score": int(prof.get("credit_score", 0) or 0),
            "loan_amount": float(prof.get("loan_amount", 0) or 0),
            "emi_amount": float(prof.get("emi_amount", 0) or 0),
            "credit_limit": float(prof.get("credit_limit", 0) or 0),
            "lgbm_prob": float(pred.get("lgbm_prob", 0)),
            "gru_prob": float(pred.get("gru_prob", 0)),  # LSTM prob
            "ensemble_prob": float(pred.get("ensemble_prob", 0)),
            "risk_score": float(pred.get("ensemble_prob", pred.get("lgbm_prob", 0))),
            "risk_level": str(pred.get("risk_level", "LOW")),
            "anomaly_flag": bool(pred.get("anomaly_flag", False)),
            "shap_top3": pred.get("shap_top3", []),
            "all_shap": pred.get("all_shap", []),
            "shap_values": pred.get("shap_values", {}),
            "human_explanation": pred.get("human_explanation", ""),
            "confidence": float(pred.get("confidence", 0.0)),
        }
        scored.append(record)

    scored = sorted(scored, key=lambda r: float(r.get("ensemble_prob", 0.0)), reverse=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    print(f"\n✅ SUCCESS: Scored {len(scored)} customers.")
    print(f"📄 Saved: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
