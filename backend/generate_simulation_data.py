"""
Generate 1000-row Simulation Dataset
=====================================
Uses the exact same 16 features as the trained models (from model_config.yaml)
but generates COMPLETELY NEW synthetic data — different distributions, random seed 999.
This data is NEVER seen during training.

Run: python generate_simulation_data.py
Output: backend/data/simulation_stream.csv
"""

import os
import numpy as np
import pandas as pd
import random

# Different seed from training data (training used seed 42)
random.seed(999)
np.random.seed(999)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(ROOT, "data", "simulation_stream.csv")

N = 1000

# Customer segments: 0=salaried, 1=self-employed, 2=farmer, 3=freelancer, 4=student, 5=other
SEGMENT_MAP = [0, 1, 2, 3, 4, 5]
SEGMENT_NAMES = ["salaried", "self-employed", "farmer", "freelancer", "student", "other"]

def generate_row(i):
    """Generate a single synthetic customer behavioral row."""
    # Randomly assign risk profile to get diverse distribution
    risk_profile = np.random.choice(["low", "medium", "high"], p=[0.45, 0.35, 0.20])
    segment_code = random.choice(SEGMENT_MAP)

    if risk_profile == "high":
        salary_delay_days          = np.random.uniform(5, 30)
        savings_wow_delta_pct      = np.random.uniform(-40, -10)
        atm_withdrawal_count_7d    = np.random.randint(6, 15)
        atm_withdrawal_amount_7d   = np.random.uniform(5000, 25000)
        discretionary_spend_7d     = np.random.uniform(4000, 15000)
        lending_upi_count_7d       = np.random.randint(3, 12)
        lending_upi_amount_7d      = np.random.uniform(3000, 20000)
        failed_autodebit_count     = np.random.randint(2, 6)
        utility_payment_delay_days = np.random.uniform(5, 20)
        gambling_spend_7d          = np.random.uniform(500, 5000)
        credit_utilization         = np.random.uniform(0.75, 0.99)
        net_cashflow_7d            = np.random.uniform(-15000, -2000)
        round_number_withdrawal_count_7d = np.random.randint(3, 10)
        weekend_spend_ratio        = np.random.uniform(0.55, 0.85)
        net_cashflow_trend_slope   = np.random.uniform(-500, -100)

    elif risk_profile == "medium":
        salary_delay_days          = np.random.uniform(1, 7)
        savings_wow_delta_pct      = np.random.uniform(-15, 5)
        atm_withdrawal_count_7d    = np.random.randint(2, 7)
        atm_withdrawal_amount_7d   = np.random.uniform(1000, 8000)
        discretionary_spend_7d     = np.random.uniform(1500, 6000)
        lending_upi_count_7d       = np.random.randint(1, 4)
        lending_upi_amount_7d      = np.random.uniform(500, 5000)
        failed_autodebit_count     = np.random.randint(0, 2)
        utility_payment_delay_days = np.random.uniform(0, 7)
        gambling_spend_7d          = np.random.uniform(0, 800)
        credit_utilization         = np.random.uniform(0.40, 0.75)
        net_cashflow_7d            = np.random.uniform(-3000, 2000)
        round_number_withdrawal_count_7d = np.random.randint(1, 5)
        weekend_spend_ratio        = np.random.uniform(0.35, 0.60)
        net_cashflow_trend_slope   = np.random.uniform(-150, 50)

    else:  # low
        salary_delay_days          = np.random.uniform(0, 2)
        savings_wow_delta_pct      = np.random.uniform(0, 20)
        atm_withdrawal_count_7d    = np.random.randint(0, 3)
        atm_withdrawal_amount_7d   = np.random.uniform(0, 3000)
        discretionary_spend_7d     = np.random.uniform(200, 2500)
        lending_upi_count_7d       = np.random.randint(0, 2)
        lending_upi_amount_7d      = np.random.uniform(0, 1000)
        failed_autodebit_count     = 0
        utility_payment_delay_days = np.random.uniform(0, 2)
        gambling_spend_7d          = np.random.uniform(0, 200)
        credit_utilization         = np.random.uniform(0.05, 0.40)
        net_cashflow_7d            = np.random.uniform(1000, 20000)
        round_number_withdrawal_count_7d = np.random.randint(0, 3)
        weekend_spend_ratio        = np.random.uniform(0.15, 0.40)
        net_cashflow_trend_slope   = np.random.uniform(50, 400)

    # Customer identity info (not used by model — for display only)
    cid = f"SIM-{i+1:04d}"
    names = ["Amit", "Priya", "Rahul", "Sneha", "Vikram", "Pooja", "Arjun", "Meera",
             "Suresh", "Kavita", "Rajan", "Nisha", "Deepak", "Anjali", "Mohit"]
    surnames = ["Sharma", "Patel", "Verma", "Singh", "Kumar", "Gupta", "Joshi", "Mehta",
                "Nair", "Rao", "Khan", "Desai", "Iyer", "Malhotra", "Bose"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata",
              "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Chandigarh", "Nagpur", "Indore"]

    return {
        "customer_id":                   cid,
        "name":                          f"{random.choice(names)} {random.choice(surnames)}",
        "city":                          random.choice(cities),
        "customer_segment":              segment_code,
        "customer_segment_name":         SEGMENT_NAMES[segment_code],
        # ── 16 Model Features ──
        "salary_delay_days":             round(salary_delay_days, 2),
        "savings_wow_delta_pct":         round(savings_wow_delta_pct, 2),
        "atm_withdrawal_count_7d":       int(atm_withdrawal_count_7d),
        "atm_withdrawal_amount_7d":      round(atm_withdrawal_amount_7d, 2),
        "discretionary_spend_7d":        round(discretionary_spend_7d, 2),
        "lending_upi_count_7d":          int(lending_upi_count_7d),
        "lending_upi_amount_7d":         round(lending_upi_amount_7d, 2),
        "failed_autodebit_count":        int(failed_autodebit_count),
        "utility_payment_delay_days":    round(utility_payment_delay_days, 2),
        "gambling_spend_7d":             round(gambling_spend_7d, 2),
        "credit_utilization":            round(credit_utilization, 4),
        "net_cashflow_7d":               round(net_cashflow_7d, 2),
        "round_number_withdrawal_count_7d": int(round_number_withdrawal_count_7d),
        "weekend_spend_ratio":           round(weekend_spend_ratio, 4),
        "net_cashflow_trend_slope":      round(net_cashflow_trend_slope, 4),
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  SIMULATION DATA GENERATOR")
    print("  Generating 1000 rows of UNSEEN unlabeled customer data")
    print("=" * 55)

    rows = [generate_row(i) for i in range(N)]
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n- Saved {len(df)} rows -> {OUTPUT_PATH}")
    print(f"\n- Feature columns: {[c for c in df.columns if c not in ['customer_id','name','city','customer_segment_name']]}")
