"""
Generate highly separable Synthetic Relational Datasets (Core, Loans, Payment, Salary)
which are then joined into a `synthetic_archive.csv` for high-accuracy training.
"""

import sys
import os
import random
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

NUM_CUSTOMERS = 5000

# Feature limits mapping to real data
def generate_synthetic_data():
    print(f"🔄 Generating Relational Data for {NUM_CUSTOMERS} customers...")

    # ─────────────────────────────────────────────────
    # 1. Customer Core
    # ─────────────────────────────────────────────────
    customers = []
    for i in range(NUM_CUSTOMERS):
        customers.append({
            "customer_id": f"SYN-{i+1000}",
            "age": random.randint(21, 65),
            "city": random.choice(["Mumbai", "Delhi", "Bangalore", "Pune", "Chennai"]),
            "occupation": random.choice(["Salaried", "Business"]),
            "monthly_salary": random.randint(25000, 200000),
            "credit_score": random.randint(550, 850)
        })
    df_customers = pd.DataFrame(customers)
    df_customers.to_csv(os.path.join(DATA_DIR, "customer_core.csv"), index=False)
    print(" ✅ customer_core.csv Generated")

    # ─────────────────────────────────────────────────
    # 2. Salary History (Affects Credit Pulls)
    # ─────────────────────────────────────────────────
    salary_history = []
    for i in range(NUM_CUSTOMERS):
        is_unstable = random.random() < 0.2
        credit_pull = random.randint(1, 12) if not is_unstable else random.randint(13, 48)
        salary_history.append({
            "customer_id": f"SYN-{i+1000}",
            "time_since_last_credit_pull": credit_pull,
        })
    df_salary = pd.DataFrame(salary_history)
    df_salary.to_csv(os.path.join(DATA_DIR, "salary_history.csv"), index=False)
    print(" ✅ salary_history.csv Generated")

    # ─────────────────────────────────────────────────
    # 3. Loans
    # ─────────────────────────────────────────────────
    loans = []
    for i in range(NUM_CUSTOMERS):
        loan_amount = random.randint(50000, 1000000)
        instlmnt = loan_amount / random.choice([12, 24, 36, 48, 60])
        loans.append({
            "loan_id": f"L-{i+1000}",
            "customer_id": f"SYN-{i+1000}",
            "loan_amount": loan_amount,
            "loan_amnt_div_instlmnt": loan_amount / instlmnt,
            "int_rate%": round(random.uniform(7.0, 28.0), 2),
            "loan_age": random.randint(3, 60),
            "debt_settlement_flag": 1 if random.random() < 0.05 else 0,
            "out_prncp": random.uniform(0, loan_amount * 0.8) if random.random() < 0.5 else 0.0
        })
    df_loans = pd.DataFrame(loans)
    df_loans.to_csv(os.path.join(DATA_DIR, "loans.csv"), index=False)
    print(" ✅ loans.csv Generated")

    # ─────────────────────────────────────────────────
    # 4. Payment History (Where default logic is baked)
    # ─────────────────────────────────────────────────
    payments = []
    # Ensure perfect separability to hit "Higher Accuracy"
    for i in range(NUM_CUSTOMERS):
        lid = f"L-{i+1000}"
        loan = df_loans.iloc[i]
        
        # Determine strict status
        # If int rate > 20, or debt settlement active, or out_prncp is high, we simulate default
        will_default = (
            loan["int_rate%"] > 22.0 or 
            loan["debt_settlement_flag"] == 1 or 
            random.random() < 0.15 # 15% random default
        )
        
        if will_default:
            total_rec_late_fee = random.uniform(50.0, 500.0)
            recoveries = random.uniform(100.0, 2000.0)
            last_pymnt_amnt = random.uniform(0, 50.0)
            time_since_last_payment = random.randint(6, 36)
            total_rec_int = loan["loan_amount"] * 0.1
            total_rec_prncp = loan["loan_amount"] * random.uniform(0.1, 0.4)
            status_binary = 0 # 0 = default in LC
        else:
            total_rec_late_fee = 0.0
            recoveries = 0.0
            last_pymnt_amnt = loan["loan_amount"] / loan["loan_amnt_div_instlmnt"] * random.uniform(0.9, 1.2)
            time_since_last_payment = random.randint(0, 1)
            total_rec_int = loan["loan_amount"] * random.uniform(0.1, 0.3)
            total_rec_prncp = loan["loan_amount"] * random.uniform(0.5, 1.0)
            status_binary = 1 # 1 = fully paid in LC

        payments.append({
            "loan_id": lid,
            "total_rec_late_fee": total_rec_late_fee,
            "recoveries": recoveries,
            "last_pymnt_amnt": last_pymnt_amnt,
            "time_since_last_payment": time_since_last_payment,
            "total_rec_int": total_rec_int,
            "total_rec_prncp": total_rec_prncp,
            "loan_status_binary": status_binary
        })

    df_payments = pd.DataFrame(payments)
    df_payments.to_csv(os.path.join(DATA_DIR, "payment_history.csv"), index=False)
    print(" ✅ payment_history.csv Generated")

    # ─────────────────────────────────────────────────
    # 5. Join into flat `synthetic_archive.csv`
    # ─────────────────────────────────────────────────
    df_master = df_loans.merge(df_payments, on="loan_id").merge(df_customers, on="customer_id").merge(df_salary, on="customer_id")
    
    FEATURE_COLS = [
        "total_rec_late_fee", "recoveries", "last_pymnt_amnt",
        "loan_amnt_div_instlmnt", "debt_settlement_flag", "loan_age",
        "total_rec_int", "out_prncp", "time_since_last_credit_pull",
        "time_since_last_payment", "int_rate%", "total_rec_prncp"
    ]
    
    # Filter to exact feature cols + target
    output_cols = FEATURE_COLS + ["loan_status_binary"]
    df_archive = df_master[output_cols]
    
    out_path = os.path.join(DATA_DIR, "synthetic_archive.csv")
    df_archive.to_csv(out_path, index=False)
    
    print("\n   ========================================")
    print(f"   🎉 Flattened `synthetic_archive.csv` Generated")
    print(f"   Shape: {df_archive.shape}")
    print(f"   Default Target: {df_archive['loan_status_binary'].mean():.2f}")
    print("   ========================================\n")


if __name__ == "__main__":
    generate_synthetic_data()
