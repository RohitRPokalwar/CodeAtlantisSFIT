"""
Risk Predictor Module — Final Architecture Models
Uses ONLY:
  - LightGBM (behavioral risk agent)
  - LSTM (time trends)
  - XGBoost (financial strength; used when customer profile is available)

Target: will_default_next_30d
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yaml
import os
import sys
import json
from inference.ai_explain import generate_ai_explanation

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _pick_id_col(df: pd.DataFrame) -> str:
    return "user_id" if "user_id" in df.columns else "customer_id"


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.25):
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

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class RiskPredictor:
    """Risk prediction using the Final Architecture models."""

    FEATURE_COLS = [
        "salary_delay_days", "savings_wow_delta_pct", "atm_withdrawal_count_7d",
        "atm_withdrawal_amount_7d", "discretionary_spend_7d", "lending_upi_count_7d",
        "lending_upi_amount_7d", "failed_autodebit_count", "utility_payment_delay_days",
        "gambling_spend_7d", "credit_utilization", "net_cashflow_7d"
    ]

    FEATURE_DESCRIPTIONS = {
        "salary_delay_days": "salary delayed by {value:.0f} days",
        "savings_wow_delta_pct": "savings changed {value:.1f}% week-over-week",
        "atm_withdrawal_count_7d": "{value:.0f} ATM withdrawals in 7 days",
        "atm_withdrawal_amount_7d": "ATM withdrawal amount: ₹{value:.0f}",
        "discretionary_spend_7d": "discretionary spending: ₹{value:.0f}",
        "lending_upi_count_7d": "{value:.0f} UPI transfers to lending apps",
        "lending_upi_amount_7d": "lending app UPI amount: ₹{value:.0f}",
        "failed_autodebit_count": "{value:.0f} failed auto-debits",
        "utility_payment_delay_days": "utility payments delayed {value:.0f} days",
        "gambling_spend_7d": "gambling spend: ₹{value:.0f}",
        "credit_utilization": "credit utilization: {value:.1%}",
        "net_cashflow_7d": "net cashflow: ₹{value:.0f}",
    }

    def __init__(self):
        with open(os.path.join(ROOT, "config", "model_config.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        with open(os.path.join(ROOT, "config", "thresholds.yaml"), "r") as f:
            self.thresholds = yaml.safe_load(f)

        self.features = self.config["features"]["tabular"]
        self.seq_features = self.config["features"]["sequence"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models_dir = os.path.join(ROOT, "models")

        # Load models
        self.lgbm = joblib.load(os.path.join(models_dir, "lgbm_model.pkl"))
        try:
            self.xgb = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
        except Exception:
            self.xgb = None

        # LSTM artifacts
        self.lstm_scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
        ckpt = torch.load(os.path.join(models_dir, "lstm_model.pt"), map_location=self.device)
        lstm_cfg = self.config.get("lstm", {})
        self.lstm = LSTMModel(
            input_size=len(self.seq_features),
            hidden_size=int(lstm_cfg.get("hidden_size", 64)),
            num_layers=int(lstm_cfg.get("num_layers", 1)),
            dropout=float(lstm_cfg.get("dropout", 0.25)),
        ).to(self.device)
        self.lstm.load_state_dict(ckpt["state_dict"])
        self.lstm.eval()

        # SHAP explainer (lazy-loaded)
        self._shap_explainer = None

        # Load behavioral data (for customer-based lookups)
        try:
            self.weekly_df = pd.read_csv(os.path.join(ROOT, "data", "weekly_behavior.csv"))
            self.customers_df = pd.read_csv(
                os.path.join(ROOT, "data", "customers.csv"))
        except Exception:
            self.weekly_df = pd.DataFrame()
            self.customers_df = pd.DataFrame()

        # Thresholds
        rt = self.thresholds["risk_thresholds"]
        self.th_low = rt["monitor_only"]
        self.th_med = rt["low_intervention"]
        self.th_high = rt["high_risk"]

        print(f"[OK] RiskPredictor loaded - {len(self.features)} features, device={self.device}")

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.lgbm)
        return self._shap_explainer

    # ── Direct prediction from raw features ──────────────────

    def predict_from_features(self, feature_values: dict):
        """Predict behavioral risk from a single feature vector.

        This endpoint stays backward-compatible with the website UI:
        - returns keys: lgbm_prob, gru_prob, ensemble_prob, anomaly_flag, risk_level, SHAP fields
        - BUT: gru_prob is now the LSTM probability (time trend proxy)
        - ensemble_prob is a weighted blend of LightGBM + LSTM
        """
        x = np.array([[feature_values.get(f, 0.0) for f in self.features]],
                      dtype=np.float32)

        # ── LightGBM ──
        lgbm_prob = float(self.lgbm.predict(x)[0])

        # ── LSTM (simulate sequence by repeating the feature vector mapped to seq_features) ──
        seq_len = int(self.config.get("lstm", {}).get("seq_len", 8))
        seq_vec = np.array([[feature_values.get(f, 0.0) for f in self.seq_features]], dtype=np.float32)
        seq = np.tile(seq_vec, (seq_len, 1))  # (seq_len, n_features)
        seq_scaled = self.lstm_scaler.transform(seq).reshape(1, seq_len, len(self.seq_features))
        with torch.no_grad():
            logits = self.lstm(torch.tensor(seq_scaled, dtype=torch.float32, device=self.device))
            lstm_prob = float(torch.sigmoid(logits).cpu().numpy().flatten()[0])

        # Simple rule-based anomaly flag (we removed Isolation Forest)
        anomaly_flag = bool(
            float(feature_values.get("failed_autodebit_count", 0) or 0) >= 3
            or float(feature_values.get("credit_utilization", 0) or 0) >= 0.92
            or float(feature_values.get("salary_delay_days", 0) or 0) >= 12
        )

        # Blend (behavior agent driven primarily by LightGBM)
        ensemble_prob = float(np.clip(0.65 * lgbm_prob + 0.35 * lstm_prob, 0.0, 1.0))

        # Risk level
        if ensemble_prob >= self.th_high:
            risk_level = "HIGH"
        elif ensemble_prob >= self.th_low:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # SHAP explanation — pass real model outputs for AI-powered narrative
        shap_result = self._compute_shap(
            x, lgbm_prob=lgbm_prob, lstm_prob=lstm_prob,
            ensemble_prob=ensemble_prob, anomaly_flag=anomaly_flag,
            risk_level=risk_level
        )

        return {
            "lgbm_prob": round(lgbm_prob, 4),
            "gru_prob": round(lstm_prob, 4),  # backward-compatible key
            "ensemble_prob": round(ensemble_prob, 4),
            "anomaly_flag": anomaly_flag,
            "risk_level": risk_level,
            "shap_top3": shap_result["top_drivers"],
            "all_shap": shap_result["all_drivers"],
            "shap_values": shap_result["shap_values"],
            "human_explanation": shap_result["human_explanation"],
            "confidence": shap_result["confidence"]
        }

    def _compute_shap(self, x, lgbm_prob=0.0, lstm_prob=0.0, ensemble_prob=0.0, anomaly_flag=False, risk_level="LOW"):
        """Compute SHAP values for a single sample."""
        try:
            raw_shap = self.shap_explainer.shap_values(x)
            # CRITICAL: SHAP BUG FIX
            if isinstance(raw_shap, list):
                shap_vals = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            else:
                shap_vals = raw_shap
            shap_vals = shap_vals.flatten()
        except Exception as e:
            print(f"SHAP error: {e}")
            shap_vals = np.zeros(len(self.features))

        # Confidence Calculation
        confidence = 1.0 - np.var([lgbm_prob, lstm_prob, ensemble_prob])
        confidence = round(np.clip(confidence, 0.4, 0.99), 2)

        feature_contribs = []
        for i, fname in enumerate(self.features):
            feature_contribs.append({
                "feature": fname,
                "contribution": round(float(shap_vals[i]), 4),
                "direction": "INCREASES_RISK" if shap_vals[i] > 0 else "DECREASES_RISK",
                "abs_contribution": abs(float(shap_vals[i]))
            })
        feature_contribs.sort(key=lambda c: c["abs_contribution"], reverse=True)

        top_drivers = [{
            "feature": fc["feature"],
            "contribution": fc["contribution"],
            "direction": fc["direction"]
        } for fc in feature_contribs[:3]]

        # Build feature values dict for the explainer
        fv = {f: float(x.flatten()[i]) for i, f in enumerate(self.features)}

        # Generate AI-powered explanation using real model data
        explanation = generate_ai_explanation(
            shap_drivers=feature_contribs,
            feature_values=fv,
            ensemble_prob=ensemble_prob,
            lgbm_prob=lgbm_prob,
            gru_prob=lstm_prob,  # keep key name for downstream prompts/templates
            anomaly_flag=anomaly_flag,
            risk_level=risk_level,
        )

        return {
            "shap_values": {f: round(float(v), 4) for f, v in zip(self.features, shap_vals)},
            "top_drivers": top_drivers,
            "all_drivers": feature_contribs,
            "human_explanation": explanation,
            "confidence": confidence
        }

    # _generate_explanation is now handled by inference.ai_explain module
    # which provides both Gemini-powered and improved template-based explanations

    # ── Customer-based prediction (uses behavioral CSVs) ──────

    def predict_single(self, customer_id, week_number=None):
        """Predict risk for a customer from behavioral data.

        Falls back to CSV-based risk scores if models can't match features.
        """
        if self.weekly_df.empty:
            return {"error": "No behavioral data loaded"}

        id_col = _pick_id_col(self.weekly_df)
        cust_data = self.weekly_df[self.weekly_df[id_col].astype(str) == str(customer_id)]
        if len(cust_data) == 0:
            return {"error": f"Customer {customer_id} not found"}

        if week_number is None:
            week_number = int(cust_data["week_number"].max())

        latest = cust_data[cust_data["week_number"] == week_number]
        if len(latest) == 0:
            latest = cust_data[cust_data["week_number"] == cust_data["week_number"].max()]
        latest_row = latest.iloc[0]

        # The behavioral CSV has different features than the trained models.
        # Use the raw risk_score from the CSV and run SHAP on available features.
        risk_score = float(latest_row.get("risk_score", 0.5))

        # Map behavioral features to model features where possible
        feature_map = {}
        for f in self.features:
            if f in latest_row.index:
                feature_map[f] = float(latest_row[f])
            else:
                feature_map[f] = 0.0

        # Check if we have enough real features to run the model
        available = sum(1 for f in self.features if f in latest_row.index)

        if available >= 6:
            # Enough features overlap — run full model
            pred = self.predict_from_features(feature_map)
        else:
            # Not enough overlap — use CSV risk_score with synthetic model outputs
            pred = {
                "lgbm_prob": risk_score,
                "gru_prob": risk_score,
                "ensemble_prob": risk_score,
                "anomaly_flag": risk_score >= 0.70,
                "risk_level": "HIGH" if risk_score >= self.th_high
                              else "MEDIUM" if risk_score >= self.th_low
                              else "LOW",
                "shap_top3": [],
                "all_shap": [],
                "shap_values": {},
                "human_explanation": f"Risk score from behavioral data: {risk_score:.4f}",
            }

        # Add customer profile
        cust_profile = self.customers_df[
            self.customers_df[_pick_id_col(self.customers_df)].astype(str) == str(customer_id)]
        profile_dict = cust_profile.iloc[0].to_dict() if len(cust_profile) > 0 else {}

        return {
            "customer_id": customer_id,
            "week_number": int(week_number),
            **pred,
            "customer_profile": profile_dict,
        }

    def batch_predict(self, week_number=52):
        """Run predict_single for all customers."""
        if self.customers_df.empty:
            return pd.DataFrame()

        customer_ids = self.customers_df["customer_id"].unique()
        results = []
        for i, cid in enumerate(customer_ids):
            if (i + 1) % 200 == 0:
                print(f"  Predicting {i+1}/{len(customer_ids)}...")
            try:
                result = self.predict_single(cid, week_number)
                results.append(result)
            except Exception as e:
                results.append({
                    "customer_id": cid,
                    "ensemble_prob": 0.0,
                    "risk_level": "LOW",
                    "error": str(e)
                })

        results_df = pd.DataFrame(results)
        if "ensemble_prob" in results_df.columns:
            results_df = results_df.sort_values("ensemble_prob", ascending=False)
        return results_df


if __name__ == "__main__":
    predictor = RiskPredictor()

    # Demo: predict from raw features
    sample = {
        "total_rec_late_fee": 0.0,
        "recoveries": 0.0,
        "last_pymnt_amnt": 357.48,
        "loan_amnt_div_instlmnt": 36.1,
        "debt_settlement_flag": 0,
        "loan_age": 36,
        "total_rec_int": 2214.92,
        "out_prncp": 0.0,
        "time_since_last_credit_pull": 1,
        "time_since_last_payment": 1,
        "int_rate%": 13.56,
        "total_rec_prncp": 10000.0,
    }
    result = predictor.predict_from_features(sample)
    print("\n🔍 Prediction from raw features:")
    print(json.dumps(result, indent=2, default=str))

    # Demo: customer-based prediction (if CSV data exists)
    if not predictor.weekly_df.empty:
        cid = predictor.customers_df["customer_id"].iloc[0]
        result = predictor.predict_single(cid)
        print(f"\n🔍 Customer prediction for {cid}:")
        print(json.dumps({k: v for k, v in result.items()
                          if k != "customer_profile"}, indent=2, default=str))
