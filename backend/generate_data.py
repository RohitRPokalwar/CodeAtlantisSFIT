"""
Dataset generator entrypoint (Final Architecture).

This repo now uses the 5-layer "Final Data Architecture":
  - customers.csv
  - transactions.csv
  - salary.csv
  - payments.csv
  - weekly_behavior.csv

Run:
  python backend/generate_data.py
"""

import os
from generate_final_architecture_data import generate


if __name__ == "__main__":
    out = generate(num_users=int(os.environ.get("NUM_USERS", "15000")))
    print("\nFINAL ARCHITECTURE DATASET READY FOR TRAINING.")
    for k, v in out.items():
        print(f"  - {k}: {v}")
