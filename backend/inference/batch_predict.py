"""
Batch Prediction Module
Score all 2000 customers for the latest week.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def batch_score(week_number=52, output_path=None):
    """Run batch prediction for all customers."""
    from inference.predict import RiskPredictor
    
    print(f"  Running batch prediction for week {week_number}...")
    predictor = RiskPredictor()
    results = predictor.batch_predict(week_number)
    
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"  Saved results to {output_path}")
    
    # Summary
    high = len(results[results.get("risk_level", "") == "HIGH"])
    medium = len(results[results.get("risk_level", "") == "MEDIUM"])
    low = len(results[results.get("risk_level", "") == "LOW"])
    print(f"  Results: {high} HIGH, {medium} MEDIUM, {low} LOW risk")
    
    return results


if __name__ == "__main__":
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 52
    batch_score(week, output_path=os.path.join(ROOT, "reports", "batch_predictions.csv"))
