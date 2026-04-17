"""
AI-Powered Explanation Generator
Uses Google Gemini API to generate real, dynamic, context-aware
risk narratives based on actual SHAP values and model outputs.
Falls back to template-based explanations if no API key is set.
"""

import os
import json
from dotenv import load_dotenv

# Try to load API key from environment
def get_gemini_key():
    load_dotenv(override=True)
    return os.environ.get("GEMINI_API_KEY", "")

def generate_ai_explanation(
    shap_drivers: list,
    feature_values: dict,
    ensemble_prob: float,
    lgbm_prob: float,
    gru_prob: float,
    anomaly_flag: bool,
    risk_level: str,
) -> str:
    """Generate a rich, AI-powered explanation using Gemini.
    
    Args:
        shap_drivers: List of dicts with 'feature', 'contribution', 'direction'
        feature_values: Dict of feature name -> raw value
        ensemble_prob: Final ensemble risk probability
        lgbm_prob: LightGBM probability
        gru_prob: GRU probability
        anomaly_flag: Whether isolation forest flagged anomaly
        risk_level: HIGH / MEDIUM / LOW
        
    Returns:
        Human-readable explanation string
    """
    api_key = get_gemini_key()
    if not api_key:
        return _template_explanation(shap_drivers, feature_values, ensemble_prob, 
                                      lgbm_prob, gru_prob, anomaly_flag, risk_level)
    
    try:
        return _gemini_explanation(api_key, shap_drivers, feature_values, ensemble_prob,
                                    lgbm_prob, gru_prob, anomaly_flag, risk_level)
    except Exception as e:
        print(f"[AI Explain] Gemini API error: {e}, falling back to template")
        return _template_explanation(shap_drivers, feature_values, ensemble_prob,
                                      lgbm_prob, gru_prob, anomaly_flag, risk_level)


def _gemini_explanation(api_key, shap_drivers, feature_values, ensemble_prob, 
                         lgbm_prob, gru_prob, anomaly_flag, risk_level):
    """Call Gemini API for real AI explanation."""
    import httpx
    
    # Build structured context for Gemini
    top_3 = shap_drivers[:3] if shap_drivers else []
    shap_summary = "\n".join([
        f"  - {d['feature'].replace('_', ' ')}: SHAP={d['contribution']:.4f} ({d['direction']})"
        for d in shap_drivers[:6]
    ])
    
    feature_summary = "\n".join([
        f"  - {k.replace('_', ' ')}: {v}"
        for k, v in feature_values.items() if v != 0
    ])
    
    prompt = f"""You are a senior banking risk analyst AI. Analyze this loan's risk assessment and write a concise, professional risk narrative (3-5 sentences). 

REAL MODEL OUTPUTS (from trained ensemble of LightGBM, GRU Neural Network, and Isolation Forest):
- Ensemble Risk Score: {ensemble_prob*100:.1f}% ({risk_level} RISK)
- LightGBM Score: {lgbm_prob*100:.1f}%
- GRU Neural Net Score: {gru_prob*100:.1f}%  
- Anomaly Detection: {"ANOMALY DETECTED" if anomaly_flag else "Normal"}

TOP SHAP FEATURE ATTRIBUTIONS (real SHAP values from the model):
{shap_summary}

RAW LOAN FEATURES:
{feature_summary}

Write a natural, insightful risk narrative explaining WHY this loan is {risk_level} risk. Reference specific features and their SHAP contributions. Be specific about dollar amounts, percentages, and time periods. If anomaly was detected, mention it. Do NOT use bullet points or headers — write flowing prose. Do NOT mention "SHAP" or technical terms — explain in business language a bank manager would understand."""

    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 300,
                "topP": 0.9,
            }
        },
        timeout=15.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Gemini API returned {response.status_code}: {response.text[:200]}")
    
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return text.strip()


def _template_explanation(shap_drivers, feature_values, ensemble_prob,
                           lgbm_prob, gru_prob, anomaly_flag, risk_level):
    """Improved template-based explanation using real model data."""
    
    FEATURE_NAMES = {
        "total_rec_late_fee": "late fees",
        "recoveries": "post-charge-off recoveries",
        "last_pymnt_amnt": "last payment amount",
        "loan_amnt_div_instlmnt": "loan-to-installment ratio",
        "debt_settlement_flag": "debt settlement status",
        "loan_age": "loan age",
        "total_rec_int": "total interest received",
        "out_prncp": "outstanding principal",
        "time_since_last_credit_pull": "time since last credit pull",
        "time_since_last_payment": "time since last payment",
        "int_rate%": "interest rate",
        "total_rec_prncp": "total principal received",
    }
    
    score_pct = ensemble_prob * 100
    
    # Build the narrative
    parts = []
    
    # Opening — risk assessment with real scores
    if risk_level == "HIGH":
        parts.append(f"This loan presents significant default risk with an ensemble score of {score_pct:.1f}%.")
    elif risk_level == "MEDIUM":
        parts.append(f"This loan shows moderate risk indicators with an ensemble score of {score_pct:.1f}%, warranting closer monitoring.")
    else:
        parts.append(f"This loan is performing well with a low ensemble score of {score_pct:.1f}%.")
    
    # Model agreement analysis
    lgbm_pct = lgbm_prob * 100
    gru_pct = gru_prob * 100
    if abs(lgbm_pct - gru_pct) > 15:
        parts.append(f"The LightGBM model ({lgbm_pct:.1f}%) and GRU neural network ({gru_pct:.1f}%) show notable divergence, suggesting complex temporal patterns in the borrower's behavior.")
    else:
        parts.append(f"Both the LightGBM ({lgbm_pct:.1f}%) and GRU ({gru_pct:.1f}%) models agree on the risk assessment, reinforcing confidence in this prediction.")
    
    # Top SHAP drivers — using real values
    if shap_drivers:
        risk_drivers = [d for d in shap_drivers[:3] if d["direction"] == "INCREASES_RISK"]
        safe_drivers = [d for d in shap_drivers[:3] if d["direction"] == "DECREASES_RISK"]
        
        if risk_drivers:
            driver_descriptions = []
            for d in risk_drivers:
                fname = d["feature"]
                fval = feature_values.get(fname, 0)
                friendly = FEATURE_NAMES.get(fname, fname.replace("_", " "))
                
                if fname in ("total_rec_late_fee", "recoveries", "total_rec_int", 
                             "out_prncp", "last_pymnt_amnt", "total_rec_prncp"):
                    driver_descriptions.append(f"{friendly} (${fval:,.2f})")
                elif fname == "int_rate%":
                    driver_descriptions.append(f"{friendly} ({fval:.1f}%)")
                elif fname in ("loan_age", "time_since_last_payment", "time_since_last_credit_pull"):
                    driver_descriptions.append(f"{friendly} ({int(fval)} months)")
                elif fname == "debt_settlement_flag":
                    driver_descriptions.append("active debt settlement" if fval > 0 else friendly)
                elif fname == "loan_amnt_div_instlmnt":
                    driver_descriptions.append(f"{friendly} ({fval:.1f}x)")
                else:
                    driver_descriptions.append(f"{friendly}")
            
            parts.append(f"The primary risk drivers are: {', '.join(driver_descriptions)}.")
        
        if safe_drivers:
            safe_desc = [FEATURE_NAMES.get(d["feature"], d["feature"].replace("_", " ")) for d in safe_drivers]
            parts.append(f"Mitigating factors include favorable {' and '.join(safe_desc)}.")
    
    # Anomaly detection
    if anomaly_flag:
        parts.append("The Isolation Forest anomaly detector has flagged this loan's feature pattern as statistically unusual, suggesting behavioral deviation from the general population.")
    
    # Recommendation
    if risk_level == "HIGH":
        parts.append("Immediate review and proactive outreach are recommended.")
    elif risk_level == "MEDIUM":
        parts.append("Continued monitoring with scheduled follow-up is advised.")
    
    return " ".join(parts)
