import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Add backend to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from inference.predict import RiskPredictor

def run_simulation(interval=2.0):
    print("="*60)
    print(" 🚀 REAL-TIME RISK SCORING SIMULATION")
    print(f" Reading from: data/simulation_stream.csv")
    print(f" Interval: {interval} seconds")
    print("="*60)

    # Initialize predictor
    predictor = RiskPredictor()

    # Load simulation data
    data_path = os.path.join(ROOT, "data", "simulation_stream.csv")
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Run generate_simulation_data.py first.")
        return

    df = pd.read_csv(data_path)
    total_rows = len(df)
    
    # Path for sharing results with the API/Frontend
    output_path = os.path.join(ROOT, "data", "latest_stream_results.json")

    try:
        idx = 0
        while idx < total_rows:
            row = df.iloc[idx].to_dict()
            
            # Prepare features for the model
            features = {f: row.get(f, 0.0) for f in predictor.features}
            
            # Predict
            start_time = time.time()
            result = predictor.predict_from_features(features)
            latency = (time.time() - start_time) * 1000
            
            # Combine with identity info for display
            display_result = {
                "customer_id": row["customer_id"],
                "name": row["name"],
                "city": row["city"],
                "timestamp": time.strftime("%H:%M:%S"),
                "risk_score": result["ensemble_prob"],
                "risk_level": result["risk_level"],
                "anomaly": result["anomaly_flag"],
                "explanation": result["human_explanation"],
                "latency_ms": round(latency, 2)
            }
            
            # Print to console
            color = "\033[91m" if result["risk_level"] == "HIGH" else "\033[93m" if result["risk_level"] == "MEDIUM" else "\033[92m"
            reset = "\033[0m"
            
            print(f"[{display_result['timestamp']}] {display_result['customer_id']} | {display_result['name']} | "
                  f"Score: {color}{display_result['risk_score']:.4f}{reset} | "
                  f"Level: {color}{display_result['risk_level']}{reset} | "
                  f"Latency: {display_result['latency_ms']}ms")
            
            if result["anomaly_flag"]:
                print(f"   ⚠️  ANOMALY DETECTED for {display_result['customer_id']}")

            # Save to JSON for "Live Flagging" UI
            # We'll maintain a list of the last 10 flags
            try:
                if os.path.exists(output_path):
                    with open(output_path, "r") as f:
                        recent_flags = json.load(f)
                else:
                    recent_flags = []
            except:
                recent_flags = []
                
            recent_flags.insert(0, display_result)
            recent_flags = recent_flags[:10] # Keep last 10
            
            with open(output_path, "w") as f:
                json.dump(recent_flags, f, indent=2)
            
            idx += 1
            if idx >= total_rows:
                print("\nReached end of data. Restarting simulation...")
                idx = 0
                
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")

if __name__ == "__main__":
    run_simulation(interval=2.0)
