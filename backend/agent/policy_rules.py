"""
Policy Rules Module
Pure Python policy/eligibility checks (no LLM).
"""

import pandas as pd
import yaml
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_thresholds():
    with open(os.path.join(ROOT, "config", "thresholds.yaml"), "r") as f:
        return yaml.safe_load(f)


class PolicyChecker:
    """Rule-based policy engine for intervention eligibility."""

    def __init__(self):
        self.thresholds = load_thresholds()
        policy = self.thresholds["intervention_policy"]
        self.holiday_cooldown = policy["payment_holiday_cooldown_weeks"]
        self.max_interventions_4w = policy["max_interventions_per_4_weeks"]
        self.min_salary_rm = policy["min_salary_for_rm_call"]
        self.min_salary_premium_rm = policy["min_salary_for_premium_rm_call"]

        self.customers_df = pd.read_csv(os.path.join(ROOT, "data", "customers.csv"))
        self.intervention_log = pd.read_csv(os.path.join(ROOT, "data", "intervention_log.csv"))

    def check_eligibility(self, customer_id, week_number, risk_score):
        """Check which interventions are eligible for a customer.

        Returns:
            dict with eligibility flags and reasons
        """
        cust = self.customers_df[self.customers_df["customer_id"] == customer_id]
        if len(cust) == 0:
            return {"error": f"Customer {customer_id} not found"}

        cust = cust.iloc[0]
        cust_log = self.intervention_log[
            self.intervention_log["customer_id"] == customer_id]

        eligibility = {
            "payment_holiday": True,
            "restructuring": True,
            "financial_counseling": True,
            "rm_call": True,
            "sms_outreach": True,
            "monitor_only": True,
            "reasons": {}
        }

        # Rule 1: Check PAYMENT_HOLIDAY cooldown (last 26 weeks)
        recent_holidays = cust_log[
            (cust_log["intervention_type"] == "PAYMENT_HOLIDAY") &
            (cust_log["week_number"] >= week_number - self.holiday_cooldown)
        ]
        if len(recent_holidays) > 0:
            eligibility["payment_holiday"] = False
            eligibility["reasons"]["payment_holiday"] = (
                f"Customer received PAYMENT_HOLIDAY within last {self.holiday_cooldown} weeks"
            )

        # Rule 2: Check intervention throttling (max 2 in last 4 weeks)
        recent_interventions = cust_log[
            (cust_log["week_number"] >= week_number - 4) &
            (cust_log["intervention_type"] != "MONITOR_ONLY")
        ]
        if len(recent_interventions) >= self.max_interventions_4w:
            # Throttle all active interventions except monitoring
            eligibility["reasons"]["throttled"] = (
                f"Customer has {len(recent_interventions)} interventions in last 4 weeks"
            )

        # Rule 3: No RESTRUCTURING if no active loan
        if cust["loan_amount"] == 0:
            eligibility["restructuring"] = False
            eligibility["reasons"]["restructuring"] = "No active loan"

        # Rule 4: RM_CALL requires minimum salary
        if cust["monthly_salary"] < self.min_salary_rm:
            eligibility["rm_call"] = False
            eligibility["reasons"]["rm_call"] = (
                f"Salary below Rs.{self.min_salary_rm}"
            )

        return eligibility

    def get_recommended_intervention(self, risk_score, eligibility, shap_signals):
        """Get the recommended intervention based on risk score and eligibility.

        This is a pure rule-based fallback when LLM is unavailable.
        """
        rt = self.thresholds["risk_thresholds"]

        # Check top SHAP signals for context
        has_gambling = any(s.get("feature") == "gambling_spend_7d"
                          for s in shap_signals if s.get("direction") == "INCREASES_RISK")
        has_lending_apps = any(s.get("feature") == "lending_upi_count_7d"
                              for s in shap_signals if s.get("direction") == "INCREASES_RISK")

        if risk_score >= rt["high_risk"]:
            if eligibility.get("payment_holiday", False):
                return "PAYMENT_HOLIDAY", "CALL"
            elif eligibility.get("restructuring", False):
                return "RESTRUCTURING_OFFER", "EMAIL"
            elif eligibility.get("rm_call", False):
                return "RM_CALL", "CALL"
            else:
                return "SMS_OUTREACH", "SMS"

        elif risk_score >= rt["low_intervention"]:
            if (has_gambling or has_lending_apps) and eligibility.get("financial_counseling", True):
                return "FINANCIAL_COUNSELING", "EMAIL"
            elif eligibility.get("restructuring", False):
                return "RESTRUCTURING_OFFER", "EMAIL"
            else:
                return "SMS_OUTREACH", "SMS"

        elif risk_score >= rt["monitor_only"]:
            return "SMS_OUTREACH", "SMS"

        return "MONITOR_ONLY", "NONE"
