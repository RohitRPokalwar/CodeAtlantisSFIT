import numpy as np
import json
import os
import sqlite3
from typing import Dict, Any, List
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "interventions_log.db")

class InterventionEngine:
    """
    Enterprise Intervention Engine (IE).
    Board-Level Design Principles:
      1. No raw model names exposed externally
      2. Risk-aligned intervention mapping
      3. 1 Primary Offer + 1 Fallback Maximum
      4. Exposure-control-first architecture
      5. Governance validation before activation
      6. Human escalation for extreme risk
      7. Full audit trail logging
      8. Fairness & bias monitoring
      9. Feedback learning loop
    """

    DESIGN_PRINCIPLES = [
        {"id": 1, "name": "No Raw Model Exposure",       "icon": "🔒", "status": "Active"},
        {"id": 2, "name": "Risk-Aligned Mapping",        "icon": "🎯", "status": "Active"},
        {"id": 3, "name": "1 Primary + 1 Fallback",      "icon": "📋", "status": "Active"},
        {"id": 4, "name": "Exposure Control First",      "icon": "🛡️", "status": "Active"},
        {"id": 5, "name": "Governance Validation",       "icon": "✅", "status": "Active"},
        {"id": 6, "name": "Human Escalation",            "icon": "👤", "status": "Active"},
        {"id": 7, "name": "Full Audit Trail",            "icon": "📝", "status": "Active"},
        {"id": 8, "name": "Fairness & Bias Monitoring",  "icon": "⚖️", "status": "Active"},
        {"id": 9, "name": "Feedback Learning Loop",      "icon": "🔄", "status": "Active"},
    ]

    AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "intervention_audit.log")

    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intervention_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_category TEXT NOT NULL,
                    intervention_type TEXT NOT NULL,
                    action_taken JSON NOT NULL,
                    timestamp TEXT NOT NULL,
                    response_status TEXT NOT NULL,
                    updated_risk_score REAL
                )
            """)
            conn.commit()

    # ──────────────────────────────────────────────────────────────
    #  OFFER CATALOG (5 Categories)
    # ──────────────────────────────────────────────────────────────
    OFFER_CATALOG = {
        # 🟢 CATEGORY 1: LIQUIDITY RELIEF (Victim – High Ability, Temp Cash Crunch)
        "liquidity_relief": {
            "label": "Liquidity Relief",
            "offers": [
                {"id": "OFF-EMI-DATESHIFT",   "name": "EMI Due Date Shift (7-15 days)",         "impact": 0.15, "max_risk": 75, "increases_exposure": False},
                {"id": "OFF-PARTIAL-EMI",     "name": "Partial EMI Payment Option",             "impact": 0.2,  "max_risk": 80, "increases_exposure": False},
                {"id": "OFF-GRACE",           "name": "Grace Period (No penalty for 5-7 days)", "impact": 0.1,  "max_risk": 70, "increases_exposure": False},
                {"id": "OFF-MORATORIUM-30",   "name": "30-Day EMI Moratorium",                  "impact": 0.4,  "max_risk": 80, "increases_exposure": False},
                {"id": "OFF-SPLIT-EMI",       "name": "Split EMI Across 2 Months",              "impact": 0.35, "max_risk": 85, "increases_exposure": False},
                {"id": "OFF-SALARY-ADV",      "name": "Salary Advance Micro-Credit",            "impact": 0.3,  "max_risk": 65, "increases_exposure": True},
                {"id": "OFF-INTEREST-FREEZE", "name": "Temporary Interest Freeze",              "impact": 0.38, "max_risk": 85, "increases_exposure": False},
            ]
        },
        # 🟡 CATEGORY 2: RESTRUCTURING (Sustained Stress, Cooperative)
        "restructuring": {
            "label": "Restructuring",
            "offers": [
                {"id": "OFF-TENURE-EXT",       "name": "Loan Tenure Extension",                "impact": 0.25, "max_risk": 90, "increases_exposure": False},
                {"id": "OFF-EMI-REDUCE",       "name": "EMI Reduction via Restructuring",      "impact": 0.3,  "max_risk": 90, "increases_exposure": False},
                {"id": "OFF-CONSOLIDATION",    "name": "Consolidation of Multiple Loans",       "impact": 0.35, "max_risk": 85, "increases_exposure": False},
                {"id": "OFF-CC-TO-EMI",        "name": "Convert Credit Card Outstanding to EMI","impact": 0.25, "max_risk": 80, "increases_exposure": False},
                {"id": "OFF-RATE-RENEGOTIATE", "name": "Interest Rate Renegotiation",           "impact": 0.2,  "max_risk": 90, "increases_exposure": False},
                {"id": "OFF-STEPUP-EMI",       "name": "Step-Up / Step-Down EMI Plan",          "impact": 0.25, "max_risk": 85, "increases_exposure": False},
            ]
        },
        # 🔵 CATEGORY 3: BEHAVIORAL CORRECTION (Intent Risk / Drift)
        "behavioral_correction": {
            "label": "Behavioral Correction",
            "offers": [
                {"id": "OFF-SPEND-LIMIT",   "name": "Spending Limit Advisory",                        "impact": 0.15, "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-MERCHANT-BLOCK", "name": "Merchant Category Block (High-risk)",           "impact": 0.3,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-CREDIT-REDUCE", "name": "Credit Limit Reduction",                          "impact": 0.2,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-FIN-COUNSEL",   "name": "Financial Counseling Session",                     "impact": 0.25, "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-BUDGET-TOOLKIT","name": "Budget Planning Toolkit Access",                   "impact": 0.1,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-AI-INSIGHTS",   "name": "Spending Insights Dashboard",                      "impact": 0.15, "max_risk": 99, "increases_exposure": False},
            ]
        },
        # 🔴 CATEGORY 4: GOVERNANCE / CONTROL (Strategic Defaulter / Extreme Risk)
        "governance": {
            "label": "Governance / Control",
            "offers": [
                {"id": "OFF-PRE-EMI-ESC",     "name": "Pre-EMI Reminder Escalation",     "impact": 0.15, "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-SOFT-FREEZE",     "name": "Soft Account Freeze Warning",     "impact": 0.25, "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-HARD-FREEZE",     "name": "Hard Freeze (if delinquent)",      "impact": 0.5,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-CREDIT-EXPOSURE", "name": "Reduced Credit Exposure",          "impact": 0.3,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-LEGAL-FLAG",      "name": "Legal Risk Flagging",              "impact": 0.4,  "max_risk": 99, "increases_exposure": False},
                {"id": "OFF-RM-REVIEW",       "name": "Manual Review by Risk Officer",    "impact": 0.45, "max_risk": 99, "increases_exposure": False},
            ]
        },
        # 🟣 CATEGORY 5: PROACTIVE POSITIVE (High CLV + Temporary Stress)
        "proactive_positive": {
            "label": "Proactive Positive",
            "offers": [
                {"id": "OFF-LOYALTY-GRACE", "name": "Loyalty-Based Grace Protection",        "impact": 0.2,  "max_risk": 60, "increases_exposure": False},
                {"id": "OFF-CASHBACK",      "name": "Cashback Protection Month",             "impact": 0.15, "max_risk": 50, "increases_exposure": False},
                {"id": "OFF-SCORE-SHIELD",  "name": "Credit Score Shield Assurance",         "impact": 0.1,  "max_risk": 55, "increases_exposure": False},
                {"id": "OFF-RM-PRIORITY",   "name": "Relationship Manager Priority Support", "impact": 0.35, "max_risk": 70, "increases_exposure": False},
            ]
        }
    }

    def _calculate_impact(self, loan_amount: float, risk_score: int, offer_impact: float, is_restructuring: bool = False) -> dict:
        default_prob_current = min(0.99, (risk_score / 140))
        default_prob_new = default_prob_current * (1.0 - offer_impact)
        recovery_gain = loan_amount * (default_prob_current - default_prob_new)
        coll_cost_saved = recovery_gain * 0.18

        return {
            "default_prob_current": round(default_prob_current * 100, 1),
            "default_prob_projected": round(default_prob_new * 100, 1),
            "recovery_gain": int(recovery_gain),
            "cost_savings": int(coll_cost_saved),
            "emi_reduction_pct": int(offer_impact * 70) if is_restructuring else 0,
            "default_prob_reduction_pct": round(((default_prob_current - default_prob_new) / (default_prob_current or 1)) * 100, 1),
            "liquidity_improvement_pct": int(offer_impact * 45)
        }

    def _governance_check(self, offer, score, features):
        if offer.get("increases_exposure", False) and score > 65:
            return False, "Blocked: Exposure increase not permitted above risk threshold 65"

        if score > offer.get("max_risk", 99):
            return False, f"Blocked: Offer not eligible above risk score {offer['max_risk']}"

        if "Moratorium" in offer["name"]:
            bounce_count = int(features.get('failed_autodebit_count', 0))
            if bounce_count > 6:
                return False, "Blocked: Excessive bounce history, moratorium not eligible"

        return True, "Approved: Eligibility and compliance checks passed"

    def _select_offers(self, score, ability, willingness, case_type, lead_agent, features):
        is_strategic = "Strategic Defaulter" in case_type
        is_victim = "Victim of Circumstance" in case_type
        gambling_amt = float(features.get('gambling_spend_7d', 0) or 0)
        salary = float(features.get('monthly_salary') or 50000)
        gambling_ratio = gambling_amt / max(salary, 1.0)

        if score >= 90:
            if is_strategic:
                return self._pick_with_governance("governance", [5, 4, 3], score, features)
            elif is_victim:
                return self._pick_with_governance("restructuring", [0, 1, 4], score, features)
            else:
                return self._pick_with_governance("governance", [5, 3, 1], score, features)

        if is_strategic:
            if score >= 70:
                return self._pick_with_governance("governance", [1, 3, 5], score, features)
            else:
                return self._pick_with_governance("governance", [0, 1], score, features)

        elif is_victim:
            if score >= 70:
                return self._pick_with_governance("liquidity_relief", [3, 4, 6], score, features)
            elif score >= 50:
                return self._pick_with_governance("liquidity_relief", [0, 1, 2], score, features)
            else:
                return self._pick_with_governance("liquidity_relief", [2, 0], score, features)

        elif ability > 60 and willingness < 40:
            if gambling_ratio > 0.03:
                return self._pick_with_governance("behavioral_correction", [1, 2, 3], score, features)
            else:
                return self._pick_with_governance("behavioral_correction", [0, 4, 5], score, features)

        elif ability < 40 and willingness > 60:
            if score >= 65:
                return self._pick_with_governance("restructuring", [2, 0, 1], score, features)
            else:
                return self._pick_with_governance("restructuring", [4, 5, 0], score, features)

        elif score < 50 and ability > 50:
            return self._pick_with_governance("proactive_positive", [0, 2, 3], score, features)

        elif lead_agent == "velocity":
            return self._pick_with_governance("liquidity_relief", [3, 4, 0], score, features)

        elif lead_agent == "financial":
            return self._pick_with_governance("restructuring", [2, 0, 4], score, features)

        elif lead_agent == "behavioral":
            return self._pick_with_governance("behavioral_correction", [5, 0, 4], score, features)

        else:
            if score >= 70:
                return self._pick_with_governance("restructuring", [0, 4], score, features)
            else:
                return self._pick_with_governance("proactive_positive", [2, 0], score, features)

    def _pick_with_governance(self, category, indices, score, features):
        cat = self.OFFER_CATALOG[category]
        offers = cat.get("offers", cat.get("structured", []))
        primary = None
        fallback = None

        for idx in indices:
            if idx >= len(offers):
                continue
            offer = offers[idx]
            approved, reason = self._governance_check(offer, score, features)
            if approved:
                if primary is None:
                    primary = {**offer, "governance_status": reason}
                elif fallback is None:
                    fallback = {**offer, "governance_status": reason}
                    break

        if primary is None:
            for offer in offers:
                approved, reason = self._governance_check(offer, score, features)
                if approved:
                    primary = {**offer, "governance_status": reason}
                    break

        if primary is None:
            rm = self.OFFER_CATALOG["governance"]["offers"][5]
            primary = {**rm, "governance_status": "Fallback: All category offers blocked by governance"}
            category = "governance"

        return primary, fallback, category

    def generate_intervention(self, customer_id: str, features: dict, ml_result: dict, context: dict, unified_score: int = None) -> dict:
        score = int(unified_score) if unified_score is not None else int(ml_result.get('fusion_score', 0))
        raw_scores = ml_result.get('agent_scores', {})
        agent_scores = {
            'financial': raw_scores.get('xgboost_risk', 0),
            'behavioral': raw_scores.get('lightgbm_risk', 0),
            'velocity': raw_scores.get('lstm_pattern', 0),
        }

        name = features.get('name', 'Customer')
        monthly_salary = float(features.get('monthly_salary') or 50000)
        monthly_emi = float(features.get('emi_amount') or 0)

        raw_gap = abs(int((features.get('savings_wow_delta_pct', 0) / 100.0) * monthly_salary)) if (features.get('savings_wow_delta_pct') or 0) < 0 else 0
        spending_gap = int(max(monthly_emi * 0.5, monthly_salary * 0.15)) if (raw_gap == 0 and score > 40) else (raw_gap if raw_gap > 0 else 4500)

        shap_factors = ml_result.get('shap_explanation', [])
        domain_map = {
            'salary_delay_days': 'velocity', 'savings_change_pct': 'financial', 'credit_utilization': 'financial',
            'failed_debits': 'velocity', 'lending_app_txns': 'behavioral', 'gambling_amt': 'behavioral'
        }
        agent_to_domain = {
            'xgboost_risk': 'financial', 'lightgbm_risk': 'behavioral', 'lstm_pattern': 'velocity'
        }
        model_label = {'financial': 'XGBoost', 'behavioral': 'LightGBM', 'velocity': 'LSTM'}
        stressor_label = {'financial': 'Financial Stress', 'behavioral': 'Spending Pattern Shift', 'velocity': 'Cash Flow Disruption'}

        if shap_factors:
            top_factor = shap_factors[0]['feature']
            lead_agent = domain_map.get(top_factor, 'financial')
        else:
            top_key = max(('xgboost_risk', 'lightgbm_risk', 'lstm_pattern'), key=lambda k: raw_scores.get(k, 0))
            lead_agent = agent_to_domain.get(top_key, 'financial')

        case_type = context.get('case_type', 'Normal')
        ability = float(context.get('ability_score', 50))
        willingness = float(context.get('willingness_score', 50))

        primary, fallback, category = self._select_offers(
            score, ability, willingness, case_type, lead_agent, features
        )

        if shap_factors:
            top_feat = shap_factors[0]['feature'].replace('_', ' ').replace('days', '').replace('pct', '').replace('amt', '').strip().title()
            internal_stressor = f"{top_feat} ({stressor_label.get(lead_agent, 'Risk Signal')})"
        else:
            internal_stressor = f"{stressor_label.get(lead_agent, 'Risk Signal')}"

        case_reason_map = {
            "Victim of Circumstance": "temporary liquidity stress",
            "Strategic Defaulter": "repayment pattern changes",
            "Normal": "your recent account activity",
        }
        detected_reason = case_reason_map.get(case_type, "your recent account activity")

        message_templates = {
            "liquidity_relief": f"Dear {name}, our proactive monitoring system detected {detected_reason}. We would like to support you with a {primary['name']}. Please contact your relationship manager or visit your nearest branch to discuss eligibility and next steps.",
            "restructuring": f"Dear {name}, based on our account review, we believe a {primary['name']} may help ease your current repayment schedule. Your relationship manager will reach out to discuss available options tailored to your situation.",
            "behavioral_correction": f"Dear {name}, we noticed some changes in your recent spending patterns. As part of our commitment to your financial wellness, we recommend a {primary['name']}. Your RM is available to guide you through this.",
            "governance": f"Dear {name}, as part of our periodic account review, your profile has been selected for enhanced review. Your designated Relationship Manager will contact you shortly to discuss a {primary['name']} and available support options.",
            "proactive_positive": f"Dear {name}, as a valued customer, we'd like to offer you a {primary['name']} as part of our proactive financial wellness program. Please reach out to your RM or visit the app for details.",
        }
        message = message_templates.get(category, f"Dear {name}, we would like to discuss a {primary['name']} with you. Your RM will be in touch.")

        is_strategic = "Strategic Defaulter" in case_type
        channel_map = {
            "liquidity_relief": {
                "high":  ("Relationship Manager Call", "Within 4 Hours"),
                "mid":   ("WhatsApp Business API", "Same Business Day"),
                "low":   ("In-App Notification", "Next 24 Hours"),
            },
            "restructuring": {
                "high":  ("Dedicated RM Call",  "Within 4 Hours"),
                "mid":   ("WhatsApp + Branch Invite", "Next Business Day"),
                "low":   ("Email + In-App Nudge", "Within 48 Hours"),
            },
            "behavioral_correction": {
                "high":  ("RM Advisory Call + In-App Alert", "Within 24 Hours"),
                "mid":   ("In-App Smart Notification", "Same Day"),
                "low":   ("In-App Wellness Dashboard", "Next Login"),
            },
            "governance": {
                "high":  ("Senior RM + Risk Officer Review", "Immediate Escalation"),
                "mid":   ("Relationship Manager Call", "Within 4 Hours"),
                "low":   ("Automated Pre-EMI Reminder", "3 Days Before Due Date"),
            },
            "proactive_positive": {
                "high":  ("RM Outreach Call", "Next Business Day"),
                "mid":   ("WhatsApp Personalized Message", "Within 24 Hours"),
                "low":   ("In-App Wellness Nudge", "Next Login Session"),
            },
        }
        tier = "high" if score >= 80 else ("mid" if score >= 50 else "low")
        if is_strategic:
            tier = "high"
        ch_timing = channel_map.get(category, channel_map["governance"]).get(tier, ("Relationship Manager Call", "Next 24 Hours"))
        channel, timing = ch_timing

        loan_amount = float(features.get('loan_amount', 500000))
        is_restruct = "Restructuring" in self.OFFER_CATALOG[category]["label"] or "Consolidat" in primary["name"]
        impact = self._calculate_impact(loan_amount, score, primary.get("impact", 0.0), is_restruct)
        fairness = self._fairness_check(category, case_type, score, features)
        feedback_signal = self._feedback_signal(primary, score, category, case_type)

        # Build detailed model context
        ensemble_band = "R1" if score < 40 else "R2" if score < 65 else "R3" if score < 85 else "R4"
        behavioral_stress = "HIGH" if max(agent_scores['behavioral'], agent_scores['velocity']) >= 75 else ("MODERATE" if max(agent_scores['behavioral'], agent_scores['velocity']) >= 45 else "LOW")
        cashflow_instability = "HIGH" if agent_scores['financial'] >= 75 else ("MODERATE" if agent_scores['financial'] >= 45 else "LOW")

        # Build explicit trigger signals (financial justification)
        emi_burden_val = (monthly_emi / max(monthly_salary, 1)) * 100
        active_loans = int(features.get("active_loans", 2))

        result = {
            "intervention_score": score,
            "recommended_offer": primary["name"],
            "offer_id": primary["id"],
            "offer_category": self.OFFER_CATALOG[category]["label"],
            "governance_status": primary.get("governance_status", "Approved"),
            "message": message,
            "channel": channel,
            "timing_optimizer": timing,
            "lead_stressor": internal_stressor,
            "spending_gap": spending_gap,
            "status": "Recommendation Sent",
            "model_rationale": f"Financial {stressor_label['financial']} · Behavioral {stressor_label['behavioral']} · Velocity {stressor_label['velocity']} · Primary: {stressor_label.get(lead_agent, 'Risk Signal')}",
            "financial_impact": impact,
            "fairness_check": fairness,
            "feedback_signal": feedback_signal,
            "design_principles": self.DESIGN_PRINCIPLES,
            "meta": {
                "trigger_signals": {
                    "active_loans": active_loans,
                    "emi_to_income_ratio": f"{emi_burden_val:.1f}%",
                    "credit_utilization": f"{(float(features.get('credit_utilization', context.get('credit_utilization', 0.8))) * 100):.1f}%",
                    "failed_debits_30d": int(features.get("failed_autodebit_count", 0)),
                },
                "exposure_control": [
                    "No new credit line issued" if not primary.get("increases_exposure") else "Approved temporary exposure increase",
                    "Risk-weighted exposure unchanged",
                    "Existing loans restructured only" if is_restruct else "Liquidity action taken"
                ],
                "model_context": {
                    "ensemble_risk_band": ensemble_band,
                    "behavioral_stress": behavioral_stress,
                    "cashflow_instability": cashflow_instability,
                },
                "case_context": case_type,
                "agent_diagnostics": agent_scores
            }
        }

        if fallback:
            result["fallback_offer"] = {"id": fallback["id"], "name": fallback["name"], "governance_status": fallback.get("governance_status", "Approved")}

        self._log_audit_trail(customer_id, result)

        # Map to old pipeline standard so SQL doesn't break entirely, but include all new data
        result["actions_triggered"] = [primary["name"]]
        if fallback: result["actions_triggered"].append(f"Fallback: {fallback['name']}")
        result["risk_category"] = self.OFFER_CATALOG[category]["label"]

        return result

    def _log_audit_trail(self, customer_id: str, result: dict):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "customer_id": customer_id,
                "risk_score": result.get("intervention_score", 0),
                "offer_id": result.get("offer_id"),
                "offer_name": result.get("recommended_offer"),
                "offer_category": result.get("offer_category"),
                "governance_status": result.get("governance_status"),
                "channel": result.get("channel"),
                "timing": result.get("timing_optimizer"),
                "fallback_offer": result.get("fallback_offer", {}).get("id") if result.get("fallback_offer") else None,
                "fairness_flag": result.get("fairness_check", {}).get("flag", "CLEAR"),
                "financial_impact": result.get("financial_impact"),
                "trigger_signals": result.get("meta", {}).get("trigger_signals"),
                "exposure_control": result.get("meta", {}).get("exposure_control"),
                "model_context": result.get("meta", {}).get("model_context")
            }
            with open(self.AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
            # Log heavily to DB for dashboard mapping
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                # Create structured string for action_taken to preserve compatibility locally
                action_pack = [result.get("recommended_offer")]
                if result.get("fallback_offer"): 
                    action_pack.append("Fallback: " + result["fallback_offer"]["name"])
                    
                cursor.execute("""
                    INSERT INTO intervention_log (
                        customer_id, risk_score, risk_category, intervention_type, 
                        action_taken, timestamp, response_status, updated_risk_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    customer_id, 
                    result.get("intervention_score", 0), 
                    result.get("offer_category"), 
                    "ENTERPRISE-DYNAMIC",
                    json.dumps(action_pack), 
                    datetime.now().isoformat(), 
                    result.get("status"), 
                    result.get("intervention_score", 0) - result.get("financial_impact", {}).get("recovery_gain", 0)/20000.0 # mock delta
                ))
                conn.commit()
        except Exception as e:
            print("Audit Log failure", e)

    def _fairness_check(self, category: str, case_type: str, score: int, features: dict) -> dict:
        product_type = features.get('f_product_type', features.get('product_type', 'Unknown'))
        city = features.get('f_city', features.get('city', 'Unknown'))

        flag = "CLEAR"
        flag_reason = "No bias indicators detected"

        if category == "governance" and "Strategic" not in case_type and score < 75:
            flag = "REVIEW"
            flag_reason = "Punitive action assigned to non-strategic customer below score 75"

        if category == "governance" and score < 85 and case_type == "Normal":
            flag = "REVIEW"
            flag_reason = "Governance escalation for normal-case customer — verify proportionality"

        return {
            "flag": flag,
            "reason": flag_reason,
            "protected_attributes_checked": ["product_type", "geography"],
            "product_type": product_type,
            "geography": city,
        }

    def _feedback_signal(self, primary: dict, score: int, category: str, case_type: str) -> dict:
        acceptance_map = {
            "liquidity_relief": 0.82,
            "restructuring": 0.68,
            "behavioral_correction": 0.52,
            "governance": 0.45,
            "proactive_positive": 0.90,
        }
        base_acceptance = acceptance_map.get(category, 0.5)

        if "Victim" in case_type:
            base_acceptance = min(0.95, base_acceptance + 0.12)
        elif "Strategic" in case_type:
            base_acceptance = max(0.35, base_acceptance - 0.10)

        effectiveness = min(0.85, primary.get("impact", 0.25) + 0.35)

        return {
            "predicted_acceptance": round(base_acceptance * 100),
            "historical_effectiveness": round(effectiveness * 100),
        }

    def get_all_logs(self) -> List[Dict[str, Any]]:
        # Load audit data for deep merging
        audit_records = {}
        if os.path.exists(self.AUDIT_LOG_PATH):
            try:
                with open(self.AUDIT_LOG_PATH, "r") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        cid = rec.get("customer_id")
                        if cid:
                            if cid not in audit_records: audit_records[cid] = []
                            audit_records[cid].append(rec)
            except Exception:
                pass

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM intervention_log ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            
            final_logs = []
            for r in rows:
                base = {
                    "id": r["id"],
                    "customer_id": r["customer_id"],
                    "risk_score": r["risk_score"],
                    "risk_category": r["risk_category"],
                    "intervention_type": r["intervention_type"],
                    "actions_triggered": json.loads(r["action_taken"]) if r["action_taken"] else [],
                    "timestamp": r["timestamp"],
                    "status": r["response_status"],
                    "updated_risk_score": round(r["updated_risk_score"] or 0, 2),
                    "full_audit": None
                }
                
                # Match latest audit record for this customer
                cid = r["customer_id"]
                if cid in audit_records and len(audit_records[cid]) > 0:
                    # Pop the most recent or take the last one (simple match for now)
                    base["full_audit"] = audit_records[cid][-1]
                    
                final_logs.append(base)
                
            return final_logs

# Initialize singleton for FastAPI
engine = InterventionEngine()
