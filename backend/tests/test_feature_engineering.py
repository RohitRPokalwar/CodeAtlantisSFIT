"""Test Feature Engineering"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_feature_engineering_imports():
    from pipeline.feature_engineering import FeatureEngineer
    assert FeatureEngineer is not None

def test_feature_computation():
    from pipeline.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    result = fe.compute_weekly_features("CUS-10042", 10)
    assert "salary_delay_days" in result
    assert "net_cashflow_7d" in result
    assert result["customer_id"] == "CUS-10042"
    assert len(result) >= 12

if __name__ == "__main__":
    test_feature_engineering_imports()
    test_feature_computation()
    print("[OK] Feature engineering tests passed")
