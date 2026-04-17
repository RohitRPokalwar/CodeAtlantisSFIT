"""
Master Training Script
Runs the 3-model training pipeline in sequence:
  - LightGBM (behavioral risk agent)
  - XGBoost (financial strength)
  - LSTM (time-based trends)
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(script_path, description):
    print(f"\n{'='*70}")
    print(f"  Running: {description}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT,
        capture_output=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    if result.returncode != 0:
        print(f"  [FAIL] {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    print("=" * 70)
    print("  MASTER TRAINING PIPELINE")
    print("  Pre-Delinquency Intervention Engine - Team Code Atlantis")
    print("=" * 70)

    scripts = [
        (os.path.join(ROOT, "training", "train_lightgbm.py"), "LightGBM Training (Behavioral)"),
        (os.path.join(ROOT, "training", "train_xgboost.py"), "XGBoost Training (Financial Strength)"),
        (os.path.join(ROOT, "training", "train_lstm.py"), "LSTM Training (Time Trends)"),
    ]

    results = []
    for script_path, desc in scripts:
        success = run_script(script_path, desc)
        results.append((desc, success))

    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    for desc, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {desc}")

    # Check model artifacts
    models_dir = os.path.join(ROOT, "models")
    expected = [
        "lgbm_model.pkl",
        "xgb_model.pkl",
        "lstm_model.pt",
        "lstm_scaler.pkl",
    ]
    print(f"\n  Model artifacts in {models_dir}:")
    for f in expected:
        path = os.path.join(models_dir, f)
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"    {status} {f}")

    all_ok = all(s for _, s in results)
    if all_ok:
        print("\n  All models trained. Artifacts saved in /models/")
    else:
        print("\n  Some training steps failed. Check logs above.")


if __name__ == "__main__":
    main()
