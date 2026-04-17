"""Test API Endpoints"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api_imports():
    from api.main import app
    from api.schemas import HealthResponse, CustomerRiskSummary
    assert app is not None
    print("  API imports OK")

def test_auth_module():
    from api.auth import create_access_token, authenticate_user
    user = authenticate_user("admin", "admin123")
    assert user is not None
    assert user["username"] == "admin"
    token = create_access_token({"sub": "admin"})
    assert len(token) > 20
    print(f"  Auth: token generated ({len(token)} chars)")

def test_schemas():
    from api.schemas import OverviewMetrics
    m = OverviewMetrics(
        total_customers=2000,
        at_risk_count=412,
        high_risk_count=87,
        interventions_sent_today=156,
        recovery_rate=42.3,
        default_rate=9.7
    )
    assert m.total_customers == 2000
    print(f"  Schema validation OK")

if __name__ == "__main__":
    test_api_imports()
    test_auth_module()
    test_schemas()
    print("[OK] API tests passed")
