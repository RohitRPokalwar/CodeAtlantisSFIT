"""
Context Engine — Smart contextual feature computation for Praeventix.

Computes 6 context features per customer:
  1. user_type       — Income archetype classification
  2. season          — Current seasonal context
  3. rainfall_index  — District-level rainfall deviation (simulated)
  4. engagement_score — Digital engagement composite score
  5. income_stability — Rolling income predictability metric
  6. anomaly_type     — Categorized anomaly classification

These features adjust risk scores to prevent false positives
and improve intervention targeting accuracy.
"""

import math
import random
from datetime import datetime
from typing import Dict, Any, List, Optional


# ── Seasonal Calendar ──────────────────────────────────────
FESTIVAL_MONTHS = {10, 11}       # Diwali / Dussehra / Eid season
HARVEST_MONTHS = {3, 4, 9, 10}  # Rabi (Mar-Apr), Kharif (Sep-Oct)
TAX_MONTHS = {3}                 # Financial year end
SCHOOL_MONTHS = {4, 5, 6}       # Admission season
DRY_SEASON_MONTHS = {11, 12, 1, 2, 3}  # Agricultural dry season


def compute_user_type(customer_row: Dict[str, Any],
                      weekly_rows: List[Dict[str, Any]]) -> str:
    """Classify income archetype based on salary regularity and occupation."""
    occupation = str(customer_row.get("occupation", "")).lower()

    # Explicit occupation keywords
    agri_terms = {"farmer", "agriculture", "agri", "farming", "cultivator"}
    gig_terms = {"freelance", "gig", "contract", "self-employed", "consultant"}
    pension_terms = {"retired", "pensioner", "pension"}

    for term in agri_terms:
        if term in occupation:
            return "AGRICULTURAL"
    for term in gig_terms:
        if term in occupation:
            return "GIG_WORKER"
    for term in pension_terms:
        if term in occupation:
            return "PENSIONER"

    # Use income regularity from weekly data
    if len(weekly_rows) >= 4:
        incomes = [float(r.get("monthly_salary", r.get("net_cashflow_7d", 0)) or 0)
                   for r in weekly_rows[-12:]]
        incomes = [i for i in incomes if i > 0]
        if len(incomes) >= 3:
            mean_inc = sum(incomes) / len(incomes)
            if mean_inc > 0:
                cv = (sum((i - mean_inc) ** 2 for i in incomes) / len(incomes)) ** 0.5 / mean_inc
                if cv < 0.15:
                    return "SALARIED"
                elif cv > 0.6:
                    return "SELF_EMPLOYED"
                else:
                    return "SALARIED"

    return "SALARIED"  # default


def compute_season() -> str:
    """Determine current seasonal context."""
    month = datetime.now().month
    if month in FESTIVAL_MONTHS:
        return "FESTIVAL"
    if month in HARVEST_MONTHS:
        return "HARVEST"
    if month in TAX_MONTHS:
        return "TAX_SEASON"
    if month in SCHOOL_MONTHS:
        return "SCHOOL_ADMISSION"
    return "NORMAL"


def compute_rainfall_index(city: str) -> float:
    """Simulated district-level rainfall deviation index.
    
    In production, this would query IMD (India Meteorological Department) API.
    Range: -100 (severe drought) to +100 (flood risk).
    """
    # Simulated regional rainfall patterns
    drought_prone = {"vidarbha", "marathwada", "rayalaseema", "bundelkhand",
                     "kalahandi", "barmer", "jaisalmer"}
    flood_prone = {"assam", "bihar", "kerala", "mumbai", "chennai", "kolkata"}

    city_lower = city.lower() if city else ""

    for region in drought_prone:
        if region in city_lower:
            return round(random.uniform(-60, -20), 1)
    for region in flood_prone:
        if region in city_lower:
            return round(random.uniform(20, 60), 1)

    # Normal rainfall distribution for other regions
    month = datetime.now().month
    if month in {6, 7, 8, 9}:  # Monsoon
        return round(random.uniform(-10, 30), 1)
    elif month in DRY_SEASON_MONTHS:
        return round(random.uniform(-30, 5), 1)
    return round(random.uniform(-15, 15), 1)


def compute_engagement_score(customer_row: Dict[str, Any],
                              weekly_rows: List[Dict[str, Any]]) -> float:
    """Composite digital engagement score (0-100).
    
    Components:
    - App frequency (30%)
    - Notification read rate (25%)
    - Feature depth / transaction variety (20%)
    - Statement/account views (15%)
    - Support interactions (10%)
    """
    if not weekly_rows:
        return 50.0  # neutral default

    latest = weekly_rows[-1] if weekly_rows else {}

    # Simulate engagement from available behavioral signals
    # Active digital users have more UPI transactions, lower ATM dependency
    digital_txn_ratio = 1.0 - min(1.0,
        float(latest.get("atm_withdrawal_count_7d", 2) or 2) / 10.0)

    # Lending app usage can indicate engagement (even if stress)
    lending_activity = min(1.0,
        float(latest.get("lending_upi_count_7d", 0) or 0) / 5.0)

    # Approximate engagement components
    app_freq = digital_txn_ratio * 100
    notif_rate = max(20, digital_txn_ratio * 80 + random.uniform(-10, 10))
    feature_depth = max(10, 50 + random.uniform(-20, 20))
    statement_views = max(5, 30 + random.uniform(-15, 15))
    support_interactions = max(0, 10 + random.uniform(-10, 10))

    score = (0.30 * app_freq +
             0.25 * notif_rate +
             0.20 * feature_depth +
             0.15 * statement_views +
             0.10 * support_interactions)

    return round(max(0, min(100, score)), 1)


def compute_income_stability(weekly_rows: List[Dict[str, Any]]) -> float:
    """Rolling 6-month income predictability (0.0 to 1.0).
    
    Combines:
    - Amount consistency (1 - CV of income) — 50%
    - Timing regularity (std of salary day) — 30%
    - Source consistency — 20%
    """
    if len(weekly_rows) < 4:
        return 0.5  # neutral default

    # Use net_cashflow as income proxy if salary data not directly available
    cashflows = [float(r.get("net_cashflow_7d", 0) or 0) for r in weekly_rows[-24:]]
    positive_flows = [c for c in cashflows if c > 0]

    if len(positive_flows) < 3:
        return 0.3

    mean_cf = sum(positive_flows) / len(positive_flows)
    if mean_cf <= 0:
        return 0.2

    # Amount consistency
    variance = sum((c - mean_cf) ** 2 for c in positive_flows) / len(positive_flows)
    cv = math.sqrt(variance) / mean_cf
    amount_consistency = max(0, 1.0 - cv)

    # Timing regularity (use salary delay as proxy)
    delays = [float(r.get("salary_delay_days", 0) or 0) for r in weekly_rows[-12:]]
    if delays:
        mean_delay = sum(delays) / len(delays)
        delay_var = sum((d - mean_delay) ** 2 for d in delays) / len(delays)
        timing_reg = max(0, 1.0 - math.sqrt(delay_var) / 15.0)
    else:
        timing_reg = 0.8

    # Source consistency (approximated)
    source_consistency = 0.85 if cv < 0.3 else 0.5

    stability = (0.50 * amount_consistency +
                 0.30 * timing_reg +
                 0.20 * source_consistency)

    return round(max(0, min(1.0, stability)), 3)


def classify_anomaly_type(latest_week: Dict[str, Any],
                           risk_score: float) -> str:
    """Categorize detected anomaly into actionable type."""
    if risk_score < 0.40:
        return "NONE"

    signals_active = 0
    anomaly_type = "NONE"

    # Check individual signal categories
    spending_spike = (
        float(latest_week.get("discretionary_spend_7d", 0) or 0) > 5000 or
        float(latest_week.get("atm_withdrawal_count_7d", 0) or 0) >= 5
    )

    income_drop = (
        float(latest_week.get("salary_delay_days", 0) or 0) > 3 or
        float(latest_week.get("savings_wow_delta_pct", 0) or 0) <= -20
    )

    behavioral_shift = (
        float(latest_week.get("lending_upi_count_7d", 0) or 0) >= 2 or
        float(latest_week.get("gambling_spend_7d", 0) or 0) > 0
    )

    network_risk = False  # Would need network data

    if spending_spike:
        signals_active += 1
        anomaly_type = "SPENDING_SPIKE"
    if income_drop:
        signals_active += 1
        anomaly_type = "INCOME_DROP"
    if behavioral_shift:
        signals_active += 1
        anomaly_type = "BEHAVIORAL_SHIFT"

    if signals_active >= 3:
        return "MULTI_SIGNAL"
    if signals_active >= 1:
        return anomaly_type

    return "NONE"


# ── Edge Case Detection ────────────────────────────────────

def detect_edge_cases(customer_row: Dict[str, Any],
                       weekly_rows: List[Dict[str, Any]],
                       risk_score: float) -> List[Dict[str, Any]]:
    """Detect production-grade edge cases with actionable intelligence."""
    edge_cases = []
    if not weekly_rows:
        return edge_cases

    latest = weekly_rows[-1]

    # EC-01: Silent Risk (Ghost Customer)
    recent_activity = float(latest.get("atm_withdrawal_count_7d", 0) or 0)
    recent_upi = float(latest.get("lending_upi_count_7d", 0) or 0)
    if recent_activity <= 1 and recent_upi == 0 and len(weekly_rows) >= 4:
        prev_activity = sum(
            float(r.get("atm_withdrawal_count_7d", 0) or 0) +
            float(r.get("lending_upi_count_7d", 0) or 0)
            for r in weekly_rows[-4:-1]
        ) / 3.0
        if prev_activity > 3:
            edge_cases.append({
                "type": "SILENT_RISK",
                "name": "Ghost Customer",
                "severity": "MEDIUM",
                "description": "Activity dropped from {:.0f} to {:.0f} txns/week. Silent risk pattern detected.".format(
                    prev_activity, recent_activity + recent_upi),
                "action": "Send wellness check SMS. Schedule RM callback.",
                "confidence": 0.72
            })

    # EC-05: Fake Balance Stability
    if len(weekly_rows) >= 4:
        savings_deltas = [float(r.get("savings_wow_delta_pct", 0) or 0)
                          for r in weekly_rows[-4:]]
        volatility = max(savings_deltas) - min(savings_deltas) if savings_deltas else 0
        if volatility > 40 and abs(sum(savings_deltas) / len(savings_deltas)) < 5:
            edge_cases.append({
                "type": "FAKE_STABILITY",
                "name": "Artificial Balance Maintenance",
                "severity": "HIGH",
                "description": "High balance volatility ({:.0f}%) but near-zero net change. Possible balance manipulation.".format(volatility),
                "action": "Reclassify to AMBER. Use minimum balance for risk calc.",
                "confidence": 0.65
            })

    # EC-08: One-Time Shock
    savings_drop = float(latest.get("savings_wow_delta_pct", 0) or 0)
    salary_ok = float(latest.get("salary_delay_days", 0) or 0) <= 2
    if savings_drop <= -40 and salary_ok:
        edge_cases.append({
            "type": "ONE_TIME_SHOCK",
            "name": "Possible Emergency Expense",
            "severity": "MEDIUM",
            "description": "Savings dropped {:.0f}% but salary pattern unchanged. Likely one-time shock (medical/wedding).".format(
                abs(savings_drop)),
            "action": "Offer emergency EMI moratorium. Do NOT downgrade credit.",
            "confidence": 0.68
        })

    # EC-19: Hidden Borrowing via UPI
    lending_count = float(latest.get("lending_upi_count_7d", 0) or 0)
    lending_amount = float(latest.get("lending_upi_amount_7d", 0) or 0)
    if lending_count >= 2 and lending_amount > 10000:
        edge_cases.append({
            "type": "HIDDEN_BORROWING",
            "name": "UPI Lending App Activity",
            "severity": "HIGH",
            "description": "{:.0f} transfers to lending apps (₹{:,.0f}). Possible debt spiral.".format(
                lending_count, lending_amount),
            "action": "Immediate risk escalation. Offer debt consolidation.",
            "confidence": 0.81
        })

    # EC-18: Behavioral Drift
    if len(weekly_rows) >= 8:
        risk_scores = [float(r.get("risk_score", 0) or 0) for r in weekly_rows[-8:]]
        if len(risk_scores) >= 4:
            # Simple linear regression slope
            n = len(risk_scores)
            x_mean = (n - 1) / 2.0
            y_mean = sum(risk_scores) / n
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(risk_scores))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator > 0 else 0

            if slope > 0.02:  # Rising risk trend
                edge_cases.append({
                    "type": "BEHAVIORAL_DRIFT",
                    "name": "Gradual Risk Increase",
                    "severity": "MEDIUM",
                    "description": "Risk score trending up by {:.3f}/week over 8 weeks. Slow deterioration detected.".format(slope),
                    "action": "Proactive financial wellness check. Offer budget planning tools.",
                    "confidence": 0.74
                })

    # EC-20: Post-EMI Liquidity Crash
    emi = float(customer_row.get("emi_amount", 0) or 0)
    cashflow = float(latest.get("net_cashflow_7d", 0) or 0)
    if emi > 0 and cashflow > 0 and cashflow < emi * 0.3:
        edge_cases.append({
            "type": "POST_EMI_CRASH",
            "name": "Post-EMI Liquidity Risk",
            "severity": "HIGH",
            "description": "Weekly cashflow (₹{:,.0f}) is only {:.0f}% of EMI (₹{:,.0f}). One expense away from default.".format(
                cashflow, (cashflow / emi) * 100, emi),
            "action": "Recommend EMI date realignment. Suggest tenure extension.",
            "confidence": 0.77
        })

    # EC-04: Intervention Fatigue (check intervention count)
    intervention_count = int(customer_row.get("intervention_count", 0) or 0)
    if intervention_count >= 4:
        edge_cases.append({
            "type": "INTERVENTION_FATIGUE",
            "name": "Contact Overload",
            "severity": "LOW",
            "description": "{} interventions in recent period. Customer may be fatigued.".format(intervention_count),
            "action": "Enforce 7-day cooling period. Switch to different channel.",
            "confidence": 0.70
        })

    return edge_cases


# ── Seasonal Dampening ─────────────────────────────────────

def apply_seasonal_dampening(risk_score: float, season: str,
                              user_type: str) -> float:
    """Adjust risk score based on seasonal context to reduce false positives."""
    dampening = 0.0

    if season == "FESTIVAL":
        dampening = -0.05  # Spending spikes are normal during festivals
    elif season == "HARVEST" and user_type == "AGRICULTURAL":
        dampening = -0.08  # Agricultural income patterns
    elif season == "TAX_SEASON":
        dampening = -0.03  # Large payments for tax are normal
    elif season == "SCHOOL_ADMISSION":
        dampening = -0.04  # Education expense spikes

    adjusted = max(0.0, min(1.0, risk_score + dampening))
    return round(adjusted, 4)


# ── Master Context Computation ─────────────────────────────

def compute_full_context(customer_row: Dict[str, Any],
                          weekly_rows: List[Dict[str, Any]],
                          risk_score: float) -> Dict[str, Any]:
    """Compute all 6 context features + edge cases for a customer."""
    latest_week = weekly_rows[-1] if weekly_rows else {}
    city = str(customer_row.get("city", "Unknown"))

    user_type = compute_user_type(customer_row, weekly_rows)
    season = compute_season()
    rainfall = compute_rainfall_index(city)
    engagement = compute_engagement_score(customer_row, weekly_rows)
    stability = compute_income_stability(weekly_rows)
    anomaly = classify_anomaly_type(latest_week, risk_score)

    # Apply seasonal dampening
    adjusted_score = apply_seasonal_dampening(risk_score, season, user_type)

    # Detect edge cases
    edge_cases = detect_edge_cases(customer_row, weekly_rows, risk_score)

    return {
        "user_type": user_type,
        "season": season,
        "rainfall_index": rainfall,
        "engagement_score": engagement,
        "income_stability": stability,
        "anomaly_type": anomaly,
        "adjusted_risk_score": adjusted_score,
        "seasonal_dampening": round(adjusted_score - risk_score, 4),
        "edge_cases": edge_cases,
        "context_confidence": round(
            min(0.95, 0.5 + len(weekly_rows) * 0.04), 2)
    }
