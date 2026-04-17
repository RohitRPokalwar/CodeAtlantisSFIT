"""
Generate the "Final Data Architecture" bank dataset (15,000 users).

Outputs (in backend/data/):
  - customers.csv
  - transactions.csv
  - salary.csv
  - payments.csv
  - weekly_behavior.csv

Notes:
- Includes BOTH `user_id` (as requested) and `customer_id` (compat alias for existing app code).
- Uses bank-style rules linking monthly salary, EMI, credit limit, credit score, utilization.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


FIRST_NAMES = [
    "Amit", "Priya", "Rahul", "Neha", "Vikram", "Anita", "Suresh", "Kavita", "Arjun", "Meera",
    "Rohit", "Sneha", "Sanjay", "Pooja", "Kiran", "Divya", "Manoj", "Isha", "Nikhil", "Riya",
]
LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Verma", "Gupta", "Reddy", "Nair", "Desai", "Mehta",
    "Joshi", "Rao", "Das", "Mishra", "Iyer", "Pillai", "Agarwal", "Choudhury", "Bhat", "Menon",
]
GENDERS = ["M", "F", "O"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
OCCUPATIONS = [
    "IT Professional", "Government Employee", "Teacher", "Healthcare Worker", "Sales Executive",
    "Business Owner", "Self-Employed", "Freelancer", "Student", "Retired",
]
EMPLOYMENT_TYPES = ["salaried", "self-employed", "freelancer", "student", "retired"]
PRODUCT_TYPES = ["Personal Loan", "Home Loan", "Auto Loan", "Credit Card", "Business Loan", "Education Loan"]

CHANNELS = ["UPI", "POS", "ATM", "NEFT", "IMPS", "AUTO_DEBIT"]
SPEND_CATEGORIES = [
    "GROCERIES", "DINING", "SHOPPING", "TRAVEL", "ENTERTAINMENT",
    "ELECTRICITY", "WATER", "GAS", "BROADBAND",
    "GAMBLING_LOTTERY",
    "UPI_LENDING_APP",
]


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(x)))))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass(frozen=True)
class CustomerProfile:
    user_id: str
    name: str
    age: int
    gender: str
    city: str
    occupation: str
    employment_type: str
    monthly_salary: int
    credit_score: int
    loan_amount: int
    emi_amount: int
    credit_limit: int
    savings_balance_initial: int
    account_balance_current: int
    account_open_days: int
    product_type: str
    loan_count: int
    latent_risk: float  # 0..1 baseline risk


def _employment_type_for_occupation(occ: str) -> str:
    if occ in {"IT Professional", "Government Employee", "Teacher", "Healthcare Worker", "Sales Executive"}:
        return "salaried"
    if occ in {"Business Owner", "Self-Employed"}:
        return "self-employed"
    if occ == "Freelancer":
        return "freelancer"
    if occ == "Student":
        return "student"
    if occ == "Retired":
        return "retired"
    return random.choice(EMPLOYMENT_TYPES)


def _salary_for_employment(emp_type: str) -> int:
    # lognormal, clipped, with segment adjustments
    base = float(np.random.lognormal(mean=10.55, sigma=0.48)) / 1000.0 * 1000.0
    if emp_type == "student":
        base = random.randint(8000, 25000)
    elif emp_type == "retired":
        base = random.randint(12000, 80000)
    elif emp_type == "freelancer":
        base *= random.uniform(0.7, 1.3)
    elif emp_type == "self-employed":
        base *= random.uniform(0.9, 1.6)
    return _clip_int(base, 8000, 450000)


def _credit_score(emp_type: str, salary: int) -> int:
    # Higher salary and salaried employment tends to raise mean.
    base_mean = 710 if emp_type == "salaried" else 680
    salary_bump = np.interp(salary, [8000, 450000], [-35, 35])
    if emp_type == "student":
        base_mean = 660
        salary_bump = 0
    if emp_type == "retired":
        base_mean = 700
    score = int(np.clip(np.random.normal(base_mean + salary_bump, 75), 300, 900))
    return score


def _credit_limit(salary: int, credit_score: int) -> int:
    # Roughly 1x-8x monthly salary with score multiplier
    score_mult = np.interp(credit_score, [300, 900], [0.8, 2.0])
    raw = salary * random.uniform(1.2, 7.5) * float(score_mult)
    return _clip_int(raw, 20000, 2500000)


def _loan_and_emi(salary: int, credit_score: int, product_type: str) -> Tuple[int, int, int]:
    # Bank rule: EMI ideally <= 40-45% of salary.
    has_loan = random.random() < 0.82
    if not has_loan:
        return 0, 0, 0

    # Loan amount multiplier depends on product + credit score
    score_mult = np.interp(credit_score, [300, 900], [4.0, 45.0])
    product_mult = {
        "Personal Loan": 12.0,
        "Auto Loan": 18.0,
        "Home Loan": 60.0,
        "Credit Card": 6.0,
        "Business Loan": 30.0,
        "Education Loan": 24.0,
    }.get(product_type, 12.0)
    loan_amount = salary * random.uniform(2.0, 6.0) * (product_mult / 12.0) * float(score_mult / 20.0)
    loan_amount = _clip_int(loan_amount, 20000, 5000000)

    tenure_months = random.choice([12, 18, 24, 36, 48, 60])
    interest_rate = np.interp(credit_score, [300, 900], [0.26, 0.09]) * random.uniform(0.85, 1.15)
    monthly_rate = interest_rate / 12.0
    # EMI approx using amortization formula
    if monthly_rate <= 0:
        emi = loan_amount / tenure_months
    else:
        emi = loan_amount * (monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    emi = float(emi)
    emi_cap = salary * random.uniform(0.28, 0.45)
    emi = min(emi, emi_cap)
    emi_amount = _clip_int(emi, 0, max(0, int(salary * 0.7)))

    loan_count = 1 if product_type != "Credit Card" else random.choice([0, 1, 2])
    return loan_amount, emi_amount, loan_count


def _baseline_latent_risk(emp_type: str, credit_score: int, salary: int, emi_amount: int) -> float:
    dti = (emi_amount / salary) if salary > 0 else 0.0
    z = 0.0
    z += np.interp(credit_score, [300, 900], [1.2, -1.2])
    z += np.interp(salary, [8000, 450000], [0.6, -0.4])
    z += np.interp(dti, [0.0, 0.55], [-0.4, 1.0])
    if emp_type in {"freelancer", "self-employed"}:
        z += 0.25
    if emp_type == "student":
        z += 0.35
    if emp_type == "retired":
        z += 0.1
    # Add noise and squash
    risk = _sigmoid(z + np.random.normal(0, 0.35))
    return float(np.clip(risk, 0.02, 0.98))


def _date_for_week(base: date, week_number: int) -> date:
    return base + timedelta(days=(week_number - 1) * 7)


def _week_number_for_date(base: date, d: date) -> int:
    delta = (d - base).days
    wk = (delta // 7) + 1
    return int(max(1, min(52, wk)))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if b == 0 else (a / b)


def _remove_old_generated_csvs():
    for fname in ("customers.csv", "transactions.csv", "salary.csv", "payments.csv", "weekly_behavior.csv"):
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            os.remove(path)


def generate(num_users: int = 15000, seed: int = 42, year: int = 2025) -> Dict[str, str]:
    random.seed(seed)
    np.random.seed(seed)

    _remove_old_generated_csvs()

    base = date(year, 1, 1)

    customers_path = os.path.join(DATA_DIR, "customers.csv")
    salary_path = os.path.join(DATA_DIR, "salary.csv")
    payments_path = os.path.join(DATA_DIR, "payments.csv")
    transactions_path = os.path.join(DATA_DIR, "transactions.csv")
    weekly_path = os.path.join(DATA_DIR, "weekly_behavior.csv")

    # --- Customers ---
    customers: List[CustomerProfile] = []
    with open(customers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "user_id", "name", "age", "gender", "city", "occupation", "employment_type",
            "monthly_salary", "credit_score", "loan_amount", "emi_amount", "credit_limit",
            "savings_balance_initial", "account_balance_current", "account_open_days",
            "product_type", "loan_count",
            # compat aliases
            "customer_id",
        ])
        for i in range(num_users):
            user_id = f"USR-{100000 + i}"
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            age = random.randint(18, 70)
            gender = random.choice(GENDERS)
            city = random.choice(CITIES)
            occupation = random.choice(OCCUPATIONS)
            employment_type = _employment_type_for_occupation(occupation)
            monthly_salary = _salary_for_employment(employment_type)
            credit_score = _credit_score(employment_type, monthly_salary)
            product_type = random.choice(PRODUCT_TYPES)

            loan_amount, emi_amount, loan_count = _loan_and_emi(monthly_salary, credit_score, product_type)
            credit_limit = _credit_limit(monthly_salary, credit_score)

            savings_init = _clip_int(monthly_salary * random.uniform(0.4, 10.0), 0, 5000000)
            acct_open_days = random.randint(30, 4000)

            # Current account balance roughly savings +/- noise and recent cashflow
            acct_balance = _clip_int(savings_init * random.uniform(0.4, 1.2) + monthly_salary * random.uniform(-0.5, 1.0), 0, 8000000)

            latent_risk = _baseline_latent_risk(employment_type, credit_score, monthly_salary, emi_amount)
            customers.append(CustomerProfile(
                user_id=user_id,
                name=name,
                age=age,
                gender=gender,
                city=city,
                occupation=occupation,
                employment_type=employment_type,
                monthly_salary=monthly_salary,
                credit_score=credit_score,
                loan_amount=loan_amount,
                emi_amount=emi_amount,
                credit_limit=credit_limit,
                savings_balance_initial=savings_init,
                account_balance_current=acct_balance,
                account_open_days=acct_open_days,
                product_type=product_type,
                loan_count=loan_count,
                latent_risk=latent_risk,
            ))
            w.writerow([
                user_id, name, age, gender, city, occupation, employment_type,
                monthly_salary, credit_score, loan_amount, emi_amount, credit_limit,
                savings_init, acct_balance, acct_open_days,
                product_type, loan_count,
                user_id,  # customer_id alias
            ])

    # --- Salary (monthly) ---
    with open(salary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "date", "salary_amount", "income_type", "credit_delay_days", "income_source", "bonus_amount", "customer_id"])
        for c in customers:
            for month in range(1, 13):
                pay_day = random.randint(1, 5)
                # risk drives delays
                delay = int(np.clip(np.random.normal(loc=2 + 18 * c.latent_risk, scale=3.5), 0, 30))
                if c.employment_type == "salaried":
                    delay = min(delay, 12)
                if c.employment_type == "student":
                    delay = min(delay, 20)
                if c.employment_type == "retired":
                    delay = min(delay, 10)

                d = date(year, month, 1) + timedelta(days=pay_day - 1 + delay)
                income_type = "salary" if c.employment_type in {"salaried", "retired"} else "irregular"
                income_source = c.employment_type
                bonus = 0
                if month in {3, 10, 12} and c.employment_type == "salaried" and random.random() < 0.25:
                    bonus = _clip_int(c.monthly_salary * random.uniform(0.3, 1.2), 0, 500000)

                # amount fluctuates for non-salaried
                amt = c.monthly_salary
                if c.employment_type in {"freelancer", "self-employed"}:
                    amt = _clip_int(c.monthly_salary * random.uniform(0.6, 1.5), 0, 600000)
                if c.employment_type == "student":
                    amt = _clip_int(c.monthly_salary * random.uniform(0.85, 1.15), 0, 30000)

                w.writerow([c.user_id, d.isoformat(), amt, income_type, delay, income_source, bonus, c.user_id])

    # --- Payments (monthly EMI schedule per loan) ---
    with open(payments_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "user_id", "loan_id", "date", "emi_amount", "emi_paid", "days_late",
            "outstanding_balance", "penalty_applied", "payment_mode",
            "customer_id",
        ])
        for c in customers:
            loan_id = f"LN-{c.user_id}"
            outstanding = float(c.loan_amount)
            for month in range(1, 13):
                if c.emi_amount <= 0 or c.loan_amount <= 0:
                    break
                due = date(year, month, 5)
                # Failure chance driven by latent risk and DTI.
                dti = _safe_div(c.emi_amount, c.monthly_salary, 0.0)
                miss_prob = float(np.clip(0.03 + 0.40 * c.latent_risk + 0.45 * max(0.0, dti - 0.25), 0.01, 0.85))
                missed = random.random() < miss_prob
                if missed:
                    emi_paid = 0
                    days_late = random.randint(7, 45)
                else:
                    emi_paid = c.emi_amount if random.random() > 0.06 else int(c.emi_amount * random.uniform(0.3, 0.9))
                    days_late = 0 if emi_paid >= c.emi_amount else random.randint(1, 10)
                penalty = 1 if days_late > 0 else 0
                payment_mode = random.choice(["AUTO_DEBIT", "UPI", "NEFT"])

                outstanding = max(0.0, outstanding - float(emi_paid) * random.uniform(0.75, 1.0))
                w.writerow([
                    c.user_id, loan_id, due.isoformat(), c.emi_amount, emi_paid, days_late,
                    round(outstanding, 2), penalty, payment_mode,
                    c.user_id,
                ])

    # --- Transactions (bank behavior) ---
    # Stream-write to avoid huge memory use.
    with open(transactions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "user_id", "transaction_id", "date", "amount", "type", "category", "channel", "balance_after",
            # compat aliases for existing app
            "customer_id", "txn_id", "txn_type",
        ])
        txn_id = 0
        for c in customers:
            balance = float(c.account_balance_current)
            for month in range(1, 13):
                month_start = date(year, month, 1)

                # salary credit (once per month)
                salary_day = random.randint(1, 5)
                salary_delay = int(np.clip(np.random.normal(loc=1 + 12 * c.latent_risk, scale=3), 0, 25))
                salary_date = month_start + timedelta(days=salary_day - 1 + salary_delay)
                salary_amt = c.monthly_salary
                if c.employment_type in {"freelancer", "self-employed"}:
                    salary_amt = _clip_int(c.monthly_salary * random.uniform(0.6, 1.6), 0, 600000)
                balance += float(salary_amt)
                txn_id += 1
                w.writerow([
                    c.user_id, f"TX-{txn_id:09d}", salary_date.isoformat(), int(salary_amt),
                    "CREDIT", "SALARY", "NEFT", int(balance),
                    c.user_id, f"TXN-{txn_id:09d}", "CREDIT",
                ])

                # EMI debit (can fail)
                if c.emi_amount > 0 and c.loan_amount > 0:
                    emi_date = month_start + timedelta(days=4)
                    fail_prob = float(np.clip(0.04 + 0.35 * c.latent_risk, 0.01, 0.80))
                    failed = random.random() < fail_prob
                    txn_id += 1
                    if failed:
                        w.writerow([
                            c.user_id, f"TX-{txn_id:09d}", emi_date.isoformat(), int(c.emi_amount),
                            "FAILED", "EMI_PAYMENT", "AUTO_DEBIT", int(balance),
                            c.user_id, f"TXN-{txn_id:09d}", "FAILED",
                        ])
                    else:
                        balance = max(0.0, balance - float(c.emi_amount))
                        w.writerow([
                            c.user_id, f"TX-{txn_id:09d}", emi_date.isoformat(), int(c.emi_amount),
                            "DEBIT", "EMI_PAYMENT", "AUTO_DEBIT", int(balance),
                            c.user_id, f"TXN-{txn_id:09d}", "DEBIT",
                        ])

                # utilities (1-2)
                for _ in range(random.randint(1, 2)):
                    cat = random.choice(["ELECTRICITY", "WATER", "GAS", "BROADBAND"])
                    amt = _clip_int(c.monthly_salary * random.uniform(0.01, 0.05), 100, 20000)
                    tdate = month_start + timedelta(days=random.randint(7, 25))
                    balance = max(0.0, balance - float(amt))
                    txn_id += 1
                    w.writerow([
                        c.user_id, f"TX-{txn_id:09d}", tdate.isoformat(), amt,
                        "DEBIT", cat, random.choice(["UPI", "POS"]), int(balance),
                        c.user_id, f"TXN-{txn_id:09d}", "DEBIT",
                    ])

                # discretionary + groceries + ATM etc (6-10 per month)
                n_misc = random.randint(6, 10)
                for _ in range(n_misc):
                    cat = random.choice(SPEND_CATEGORIES)
                    chan = "ATM" if cat == "ATM_WITHDRAWAL" else random.choice(["UPI", "POS", "IMPS"])
                    if cat == "UPI_LENDING_APP":
                        amt = _clip_int(c.monthly_salary * random.uniform(0.04, 0.30) * (1.0 + 1.5 * c.latent_risk), 100, 150000)
                    elif cat == "GAMBLING_LOTTERY":
                        amt = _clip_int(c.monthly_salary * random.uniform(0.00, 0.10) * (0.6 + 1.8 * c.latent_risk), 0, 80000)
                    else:
                        amt = _clip_int(c.monthly_salary * random.uniform(0.01, 0.12), 50, 120000)

                    tdate = month_start + timedelta(days=random.randint(1, 28))
                    balance = max(0.0, balance - float(amt))
                    txn_id += 1
                    w.writerow([
                        c.user_id, f"TX-{txn_id:09d}", tdate.isoformat(), amt,
                        "DEBIT", cat, chan, int(balance),
                        c.user_id, f"TXN-{txn_id:09d}", "DEBIT",
                    ])

    # --- Weekly behavior (key for pre-delinquency) ---
    weekly_header = [
        "user_id", "week_number", "year",
        "stress_level", "salary_delay_days", "savings_balance", "savings_wow_delta_pct",
        "atm_withdrawal_count_7d", "atm_withdrawal_amount_7d",
        "discretionary_spend_7d",
        "lending_upi_count_7d", "lending_upi_amount_7d",
        "failed_autodebit_count", "utility_payment_delay_days", "gambling_spend_7d",
        "credit_utilization", "net_cashflow_7d",
        "will_default_next_30d",
        # compat alias
        "customer_id",
    ]
    with open(weekly_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(weekly_header)

        for c in customers:
            # start from initial savings and evolve
            savings = float(c.savings_balance_initial)
            prev_savings = float(savings)
            prev_util = float(np.clip(np.random.normal(0.22, 0.08), 0, 1))
            for week in range(1, 53):
                t = week / 52.0
                # stress progression: low-risk stable, high-risk worsens late-year
                drift = (c.latent_risk - 0.35) * (0.4 + 0.9 * t)
                noise = np.random.normal(0, 0.08)
                stress_score = float(np.clip(0.15 + drift + noise, 0.0, 1.0))

                salary_delay = int(np.clip(np.random.normal(1 + 20 * stress_score, 4), 0, 30))
                util = float(np.clip(prev_util + np.random.normal(0.0, 0.06) + 0.22 * (stress_score - 0.25), 0.0, 1.0))
                failed_autodebit = int((random.random() < (0.02 + 0.45 * stress_score)) and (c.emi_amount > 0))

                atm_count = int(np.clip(np.random.poisson(1 + 6 * stress_score), 0, 18))
                atm_amt = _clip_int(atm_count * random.uniform(1500, 5500), 0, 250000)
                discretionary = _clip_int(c.monthly_salary * random.uniform(0.05, 0.22) * (1.1 - 0.5 * stress_score), 0, 250000)
                lending_count = int(np.clip(np.random.poisson(0.2 + 4.0 * stress_score), 0, 20))
                lending_amt = _clip_int(lending_count * random.uniform(2000, 12000) * (0.6 + 1.4 * stress_score), 0, 300000)
                utility_delay = int(np.clip(np.random.normal(0.5 + 14 * stress_score, 3.0), 0, 45))
                gambling_spend = _clip_int(c.monthly_salary * random.uniform(0.0, 0.06) * (0.4 + 2.0 * stress_score), 0, 120000)

                # net cashflow: income-per-week minus spend (rough)
                weekly_income = c.monthly_salary / 4.0
                spend = discretionary + atm_amt + lending_amt + gambling_spend + int(c.emi_amount / 4) + int(c.monthly_salary * 0.12)
                net_cf = int(np.clip(np.random.normal(weekly_income - spend, c.monthly_salary * 0.10), -250000, 250000))

                # savings evolve
                savings = float(np.clip(savings + net_cf, 0, 15000000))
                wow_delta_pct = float(np.clip((_safe_div((savings - prev_savings), max(prev_savings, 1.0), 0.0) * 100.0), -100.0, 100.0))
                prev_savings = float(savings)
                prev_util = float(util)

                # stress_level 0/1/2 (as categorical)
                stress_level = 0 if stress_score < 0.35 else (1 if stress_score < 0.7 else 2)

                # will_default_next_30d: depends on stress signals and DTI
                dti = _safe_div(c.emi_amount, c.monthly_salary, 0.0)
                z = -3.0
                z += 2.2 * stress_score
                z += 1.2 * util
                z += 1.0 * failed_autodebit
                z += 0.8 * (salary_delay / 30.0)
                z += 0.6 * max(0.0, -net_cf / max(c.monthly_salary, 1))
                z += 0.8 * max(0.0, dti - 0.28)
                p_default = float(np.clip(_sigmoid(z), 0.01, 0.95))
                will_default = 1 if (random.random() < p_default) else 0

                w.writerow([
                    c.user_id, week, year,
                    stress_level, salary_delay, int(savings), round(wow_delta_pct, 2),
                    atm_count, atm_amt,
                    discretionary,
                    lending_count, lending_amt,
                    failed_autodebit, utility_delay, gambling_spend,
                    round(util, 4), int(net_cf),
                    will_default,
                    c.user_id,
                ])

    return {
        "customers": customers_path,
        "transactions": transactions_path,
        "salary": salary_path,
        "payments": payments_path,
        "weekly_behavior": weekly_path,
    }


if __name__ == "__main__":
    out = generate(num_users=int(os.environ.get("NUM_USERS", "15000")))
    print("✅ Generated final-architecture dataset:")
    for k, v in out.items():
        print(f"  - {k}: {v}")

