"""Test LightGBM Model"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_lgbm_model_exists():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "lgbm_model.pkl")
    if os.path.exists(model_path):
        import joblib
        model = joblib.load(model_path)
        assert model is not None
        print("  LightGBM model loaded successfully")
    else:
        print("  Model not trained yet - run training first")

def test_lgbm_config():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config", "model_config.yaml")) as f:
        config = yaml.safe_load(f)
    assert "lightgbm" in config
    assert config["lightgbm"]["objective"] == "binary"
    assert config["lightgbm"]["is_unbalance"] == True
    print("  LightGBM config validated")

if __name__ == "__main__":
    test_lgbm_config()
    test_lgbm_model_exists()
    print("[OK] LightGBM tests passed")
