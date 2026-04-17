"""Test Intervention Agent"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_policy_checker():
    from agent.policy_rules import PolicyChecker
    pc = PolicyChecker()
    result = pc.check_eligibility("CUS-10042", 52, 0.75)
    assert "payment_holiday" in result
    assert "restructuring" in result
    assert "rm_call" in result
    print(f"  Policy check: {result}")

def test_pii_masking():
    from agent.pii_masking import PIIMasker
    masked = PIIMasker.mask_customer_profile({
        "customer_id": "CUS-10042",
        "name": "Rajesh Sharma",
        "monthly_salary": 75000,
        "age": 35,
        "credit_score": 720
    })
    assert "name" not in masked
    assert "monthly_salary" not in masked
    assert masked["salary_range"] == "50k_to_100k"
    print(f"  PII masking: name removed, salary anonymised")

def test_llm_client_mock():
    from agent.llm_client import LLMClient
    client = LLMClient(mode="mock")
    response = client.invoke("system", "Risk Score: 0.82\nOther data")
    import json
    data = json.loads(response)
    assert "intervention" in data
    assert "message" in data
    print(f"  Mock LLM: {data['intervention']}")

def test_agent_run():
    from agent.intervention_agent import InterventionAgent
    agent = InterventionAgent()
    result = agent.run(
        customer_id="CUS-10042",
        week_number=52,
        risk_score=0.75,
        shap_explanations=[{"feature": "salary_delay_days", "contribution": 0.18, "direction": "INCREASES_RISK"}],
        customer_profile={"customer_id": "CUS-10042", "monthly_salary": 50000, "loan_amount": 100000}
    )
    assert result["compliance_approved"] == True
    assert result["chosen_intervention"] in ["PAYMENT_HOLIDAY", "RESTRUCTURING_OFFER", "FINANCIAL_COUNSELING", "RM_CALL", "SMS_OUTREACH"]
    # Check no aggressive words
    aggressive = ["overdue", "defaulted", "legal", "lawsuit", "court", "CIBIL", "collection agency", "debt", "delinquent"]
    msg = result["outreach_message"].lower()
    for word in aggressive:
        assert word not in msg, f"Aggressive word '{word}' found in message"
    print(f"  Agent result: {result['chosen_intervention']} via {result['chosen_channel']}")
    print(f"  Message: {result['outreach_message'][:80]}...")

if __name__ == "__main__":
    test_pii_masking()
    test_llm_client_mock()
    test_policy_checker()
    test_agent_run()
    print("[OK] Agent tests passed")
