"""
LangGraph Intervention Agent
Stateful state machine with 5 nodes for intervention decision-making.
"""

import json
import os
import sys
import yaml
import pandas as pd
from typing import TypedDict, Literal
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.llm_client import LLMClient
from agent.policy_rules import PolicyChecker
from agent.pii_masking import PIIMasker


def load_thresholds():
    with open(os.path.join(ROOT, "config", "thresholds.yaml"), "r") as f:
        return yaml.safe_load(f)


ALLOWED_INTERVENTIONS = [
    "PAYMENT_HOLIDAY", "RESTRUCTURING_OFFER", "FINANCIAL_COUNSELING",
    "RM_CALL", "SMS_OUTREACH", "MONITOR_ONLY"
]


class AgentState(TypedDict):
    customer_id: str
    week_number: int
    risk_score: float
    shap_explanations: list
    customer_profile: dict
    intervention_history: list
    eligibility: dict
    chosen_intervention: str
    chosen_channel: str
    intervention_reason: str
    outreach_message: str
    compliance_approved: bool
    dispatched: bool
    node_path: list


class InterventionAgent:
    """LangGraph-style intervention decision agent."""

    def __init__(self):
        self.llm = LLMClient()
        self.policy = PolicyChecker()
        self.pii = PIIMasker()
        self.thresholds = load_thresholds()
        self.compliance_cfg = self.thresholds["compliance"]
        self.risk_thresholds = self.thresholds["risk_thresholds"]

    def run(self, customer_id, week_number, risk_score,
            shap_explanations=None, customer_profile=None):
        """Run the full intervention pipeline."""
        state = AgentState(
            customer_id=customer_id,
            week_number=week_number,
            risk_score=risk_score,
            shap_explanations=shap_explanations or [],
            customer_profile=customer_profile or {},
            intervention_history=[],
            eligibility={},
            chosen_intervention="",
            chosen_channel="",
            intervention_reason="",
            outreach_message="",
            compliance_approved=False,
            dispatched=False,
            node_path=[]
        )

        # ── Node 1: Risk Gate ──
        state = self._risk_gate(state)
        if state["chosen_intervention"] == "MONITOR_ONLY":
            return state

        # ── Node 2: Policy Check ──
        state = self._policy_check(state)

        # ── Node 3: Decision Planner (LLM) ──
        max_retries = 2
        for attempt in range(max_retries + 1):
            state = self._decision_planner(state)

            # ── Node 4: Compliance Filter ──
            state = self._compliance_filter(state)

            if state["compliance_approved"]:
                break
            elif attempt < max_retries:
                print(f"  Compliance rejected, retry {attempt + 1}/{max_retries}...")
            else:
                # Fallback to SMS_OUTREACH
                state["chosen_intervention"] = "SMS_OUTREACH"
                state["chosen_channel"] = "SMS"
                state["outreach_message"] = (
                    "We care about your financial wellness. "
                    "Our team is here to help you with flexible options. "
                    "Please reach out to us anytime."
                )
                state["compliance_approved"] = True

        # ── Node 5: Dispatch ──
        state = self._dispatch(state)
        return state

    def _risk_gate(self, state):
        """Node 1: Route based on risk score."""
        state["node_path"].append("risk_gate")
        risk = state["risk_score"]

        if risk < self.risk_thresholds["monitor_only"]:
            state["chosen_intervention"] = "MONITOR_ONLY"
            state["chosen_channel"] = "NONE"
            state["outreach_message"] = ""
            state["compliance_approved"] = True
            state["intervention_reason"] = "Risk score below monitoring threshold"
            return state

        if risk < self.risk_thresholds["low_intervention"]:
            state["node_path"].append("low_intervention_path")
        else:
            state["node_path"].append("high_intervention_path")

        return state

    def _policy_check(self, state):
        """Node 2: Rule-based policy/eligibility checks (NO LLM)."""
        state["node_path"].append("policy_check")

        eligibility = self.policy.check_eligibility(
            state["customer_id"],
            state["week_number"],
            state["risk_score"]
        )
        state["eligibility"] = eligibility

        # Load intervention history
        log_df = pd.read_csv(os.path.join(ROOT, "data", "intervention_log.csv"))
        cust_log = log_df[log_df["customer_id"] == state["customer_id"]]
        state["intervention_history"] = cust_log.to_dict("records")

        return state

    def _decision_planner(self, state):
        """Node 3: LLM-based intervention decision."""
        state["node_path"].append("decision_planner")

        # Mask PII before LLM call
        masked_profile = self.pii.mask_customer_profile(state["customer_profile"])

        system_prompt = (
            "You are a pre-delinquency risk officer at an Indian bank. You must "
            "select the most appropriate, empathetic intervention for a financially "
            "stressed customer. You MUST respect the eligibility constraints. "
            "Respond ONLY in valid JSON."
        )

        eligibility = state["eligibility"]
        user_prompt = f"""Customer Profile: {json.dumps(masked_profile)}
Risk Score: {state['risk_score']}
SHAP Stress Signals: {json.dumps(state['shap_explanations'][:3])}
Eligible Interventions: {json.dumps(eligibility)}

Rules:
- PAYMENT_HOLIDAY: eligible={eligibility.get('payment_holiday', False)}, use if risk > 0.70
- RESTRUCTURING_OFFER: eligible={eligibility.get('restructuring', False)}, use if loan > 0 and risk > 0.65
- FINANCIAL_COUNSELING: always eligible, use if stress from gambling/lending apps
- RM_CALL: eligible={eligibility.get('rm_call', False)}, use if salary > 50000
- SMS_OUTREACH: always eligible, use as fallback

Return JSON only:
{{"intervention": "<CHOSEN_TYPE>", "channel": "<SMS|EMAIL|APP|CALL>", "reason": "<one sentence>", "message": "<empathetic outreach message max 160 chars>"}}"""

        try:
            response = self.llm.invoke(system_prompt, user_prompt)
            # Parse JSON from response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
            decision = json.loads(response_clean)

            state["chosen_intervention"] = decision.get("intervention", "SMS_OUTREACH")
            state["chosen_channel"] = decision.get("channel", "SMS")
            state["intervention_reason"] = decision.get("reason", "")
            state["outreach_message"] = decision.get("message", "")

        except (json.JSONDecodeError, Exception) as e:
            print(f"  LLM response parse error: {e}")
            # Fallback to rule-based
            intervention, channel = self.policy.get_recommended_intervention(
                state["risk_score"], state["eligibility"], state["shap_explanations"])
            state["chosen_intervention"] = intervention
            state["chosen_channel"] = channel
            state["intervention_reason"] = "Rule-based fallback"
            state["outreach_message"] = (
                "We noticed some changes in your payment patterns. "
                "We're here to help with flexible options. Please reach out."
            )

        return state

    def _compliance_filter(self, state):
        """Node 4: Compliance validation (NO LLM, rule-based)."""
        state["node_path"].append("compliance_filter")
        approved = True

        # Validate intervention is in allowed list
        if state["chosen_intervention"] not in ALLOWED_INTERVENTIONS:
            state["chosen_intervention"] = "SMS_OUTREACH"

        # Validate message length
        msg = state["outreach_message"]
        channel = state["chosen_channel"]
        max_len = self.compliance_cfg["max_sms_chars"] if channel == "SMS" else self.compliance_cfg["max_email_chars"]
        if len(msg) > max_len:
            msg = msg[:max_len - 3] + "..."
            state["outreach_message"] = msg

        # Redact PII
        customer_name = state["customer_profile"].get("name", "")
        if customer_name:
            state["outreach_message"] = self.pii.redact_name_from_message(
                state["outreach_message"], customer_name)

        # Check for aggressive words
        aggressive_words = self.compliance_cfg["aggressive_words"]
        msg_lower = state["outreach_message"].lower()
        for word in aggressive_words:
            if word.lower() in msg_lower:
                approved = False
                state["intervention_reason"] += f" [BLOCKED: aggressive word '{word}']"
                break

        state["compliance_approved"] = approved
        return state

    def _dispatch(self, state):
        """Node 5: Dispatch the intervention."""
        state["node_path"].append("dispatch")

        if not state["compliance_approved"]:
            state["dispatched"] = False
            return state

        # Append to intervention log
        if state["chosen_intervention"] != "MONITOR_ONLY":
            try:
                log_path = os.path.join(ROOT, "data", "intervention_log.csv")
                new_row = pd.DataFrame([{
                    "customer_id": state["customer_id"],
                    "week_number": state["week_number"],
                    "risk_score_at_trigger": state["risk_score"],
                    "intervention_type": state["chosen_intervention"],
                    "channel": state["chosen_channel"],
                    "status": "SENT",
                    "outcome": "PENDING",
                    "top_signal": (state["shap_explanations"][0]["feature"]
                                   if state["shap_explanations"] else "unknown")
                }])
                new_row.to_csv(log_path, mode="a", header=False, index=False)
            except Exception as e:
                print(f"  Warning: Could not write to intervention log: {e}")

        state["dispatched"] = True
        print(f"  [DISPATCHED] Customer: {state['customer_id']}")
        print(f"    Intervention: {state['chosen_intervention']}")
        print(f"    Channel: {state['chosen_channel']}")
        print(f"    Compliance: {'APPROVED' if state['compliance_approved'] else 'REJECTED'}")
        print(f"    Message: {state['outreach_message']}")
        return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer", default="CUS-10042")
    parser.add_argument("--week", type=int, default=52)
    args = parser.parse_args()

    # Get risk prediction
    try:
        from inference.predict import RiskPredictor
        predictor = RiskPredictor()
        prediction = predictor.predict_single(args.customer, args.week)
        risk_score = prediction["ensemble_prob"]
        shap_explanations = prediction["shap_top3"]
        customer_profile = prediction.get("customer_profile", {})
    except Exception as e:
        print(f"  Could not load predictor: {e}")
        risk_score = 0.75
        shap_explanations = [{"feature": "salary_delay_days", "contribution": 0.18, "direction": "INCREASES_RISK"}]
        customer_profile = {"customer_id": args.customer, "monthly_salary": 50000}

    agent = InterventionAgent()
    result = agent.run(
        customer_id=args.customer,
        week_number=args.week,
        risk_score=risk_score,
        shap_explanations=shap_explanations,
        customer_profile=customer_profile
    )
    print(f"\n  Agent path: {' -> '.join(result['node_path'])}")
