"""
Feature Engineering Module
Computes all 12 behavioral features from raw transactions.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """Computes weekly behavioral features from raw transaction data."""

    def __init__(self, transactions_path="data/transactions.csv",
                 customers_path="data/customers.csv"):
        self.transactions = pd.read_csv(transactions_path)
        self.transactions["date"] = pd.to_datetime(self.transactions["date"])
        self.customers = pd.read_csv(customers_path)
        self.customers.set_index("customer_id", inplace=True)

    def compute_weekly_features(self, customer_id, reference_week):
        """Compute all 12 features for a given customer and week.
        Uses only transactions BEFORE that week (no data leakage)."""
        cust = self.customers.loc[customer_id]
        txns = self.transactions[self.transactions["customer_id"] == customer_id].copy()

        # Convert week to approximate date range (7 days per week from Jan 1)
        base_date = pd.Timestamp("2025-01-01")
        week_end = base_date + pd.Timedelta(days=reference_week * 7)
        week_start = week_end - pd.Timedelta(days=7)
        prev_week_start = week_start - pd.Timedelta(days=7)

        # Filter transactions for the current 7-day window
        mask_7d = (txns["date"] >= week_start) & (txns["date"] < week_end)
        txns_7d = txns[mask_7d]

        # Previous 7-day window for comparison
        mask_prev = (txns["date"] >= prev_week_start) & (txns["date"] < week_start)
        txns_prev = txns[mask_prev]

        # 1. salary_delay_days
        salary_txns = txns_7d[txns_7d["category"] == "SALARY"]
        if len(salary_txns) > 0:
            salary_day = salary_txns["date"].dt.day.mean()
            salary_delay = max(0, salary_day - 5)
        else:
            salary_delay = 0

        # 2. savings_wow_delta_pct
        credits_7d = txns_7d[txns_7d["txn_type"] == "CREDIT"]["amount"].sum()
        debits_7d = txns_7d[txns_7d["txn_type"] == "DEBIT"]["amount"].sum()
        credits_prev = txns_prev[txns_prev["txn_type"] == "CREDIT"]["amount"].sum()
        debits_prev = txns_prev[txns_prev["txn_type"] == "DEBIT"]["amount"].sum()
        net_7d = credits_7d - debits_7d
        net_prev = credits_prev - debits_prev
        if abs(net_prev) > 0:
            savings_wow_delta = ((net_7d - net_prev) / abs(net_prev)) * 100
        else:
            savings_wow_delta = 0

        # 3. atm_withdrawal_count_7d
        atm_mask = txns_7d["category"] == "ATM_WITHDRAWAL"
        atm_count = atm_mask.sum()

        # 4. atm_withdrawal_amount_7d
        atm_amount = txns_7d[atm_mask]["amount"].sum()

        # 5. discretionary_spend_7d
        disc_cats = ["DINING", "ENTERTAINMENT", "SHOPPING", "TRAVEL"]
        discretionary = txns_7d[txns_7d["category"].isin(disc_cats)]["amount"].sum()

        # 6. lending_upi_count_7d
        lending_mask = txns_7d["category"] == "UPI_LENDING_APP"
        lending_count = lending_mask.sum()

        # 7. lending_upi_amount_7d
        lending_amount = txns_7d[lending_mask]["amount"].sum()

        # 8. failed_autodebit_count
        failed_count = (txns_7d["txn_type"] == "FAILED").sum()

        # 9. utility_payment_delay_days
        utility_cats = ["ELECTRICITY", "WATER", "GAS", "BROADBAND"]
        utility_txns = txns_7d[txns_7d["category"].isin(utility_cats)]
        if len(utility_txns) > 0:
            util_delay = max(0, utility_txns["date"].dt.day.mean() - 5)
        else:
            util_delay = 0

        # 10. gambling_spend_7d
        gambling = txns_7d[txns_7d["category"] == "GAMBLING_LOTTERY"]["amount"].sum()

        # 11. credit_utilization
        credit_limit = cust["credit_limit"]
        if credit_limit > 0:
            credit_util = min(debits_7d / credit_limit, 1.0)
        else:
            credit_util = 0

        # 12. net_cashflow_7d
        net_cashflow = credits_7d - debits_7d

        # 13. round_number_withdrawal_count_7d
        round_withdrawals = txns_7d[atm_mask & ((txns_7d["amount"] % 5000 == 0) | (txns_7d["amount"] % 1000 == 0))]
        round_count = len(round_withdrawals)

        # 14. weekend_spend_ratio
        disc_txns = txns_7d[txns_7d["category"].isin(disc_cats)]
        if len(disc_txns) > 0 and discretionary > 0:
            weekend_spend = disc_txns[disc_txns["date"].dt.dayofweek >= 5]["amount"].sum()
            weekend_ratio = weekend_spend / discretionary
        else:
            weekend_ratio = 0

        # 15. net_cashflow_trend_slope (4 weeks history)
        if reference_week >= 4:
            history_start = week_end - pd.Timedelta(days=28)
            mask_28d = (txns["date"] >= history_start) & (txns["date"] < week_end)
            txns_28d = txns[mask_28d]
            # Simple calculation using 1st week vs 4th week
            w1_start = history_start
            w1_end = history_start + pd.Timedelta(days=7)
            w1_txns = txns_28d[(txns_28d["date"] >= w1_start) & (txns_28d["date"] < w1_end)]
            w1_cashflow = w1_txns[w1_txns["txn_type"] == "CREDIT"]["amount"].sum() - w1_txns[w1_txns["txn_type"] == "DEBIT"]["amount"].sum()
            
            slope = (net_cashflow - w1_cashflow) / 4.0
        else:
            slope = 0.0

        customer_segment = cust.get("customer_segment", 0)

        return {
            "customer_id": customer_id,
            "week_number": reference_week,
            "salary_delay_days": round(salary_delay),
            "savings_wow_delta_pct": round(savings_wow_delta, 2),
            "atm_withdrawal_count_7d": int(atm_count),
            "atm_withdrawal_amount_7d": int(atm_amount),
            "discretionary_spend_7d": int(discretionary),
            "lending_upi_count_7d": int(lending_count),
            "lending_upi_amount_7d": int(lending_amount),
            "failed_autodebit_count": int(failed_count),
            "utility_payment_delay_days": round(util_delay),
            "gambling_spend_7d": int(gambling),
            "credit_utilization": round(credit_util, 4),
            "net_cashflow_7d": int(net_cashflow),
            "customer_segment": int(customer_segment),
            "round_number_withdrawal_count_7d": int(round_count),
            "weekend_spend_ratio": round(weekend_ratio, 4),
            "net_cashflow_trend_slope": round(slope, 4),
        }

    def batch_compute(self, output_path="data/computed_features.csv"):
        """Run compute_weekly_features for all customers and all 52 weeks."""
        all_features = []
        customer_ids = self.customers.index.tolist()
        for i, cid in enumerate(customer_ids):
            if i % 100 == 0:
                print(f"  Processing customer {i+1}/{len(customer_ids)}...")
            for week in range(1, 53):
                features = self.compute_weekly_features(cid, week)
                all_features.append(features)
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} rows to {output_path}")
        return df


if __name__ == "__main__":
    fe = FeatureEngineer()
    sample = fe.compute_weekly_features("CUS-10042", 52)
    print("Sample features for CUS-10042, week 52:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
