"""Dataset: a synthetic telecom "customer churn" table.

Why synthetic? It works offline, it is reproducible (fixed seed), and because we
*wrote* the rules that generate churn we can check whether our models and
explanations recover them. Real-world data is messier, but the workflow is the same.

Each row is one customer.
  - FEATURES (X): what we know about the customer (inputs to the model)
  - LABEL   (y): `churned` -> 1 if the customer left, 0 if they stayed (what we predict)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
RANDOM_STATE = 42

ID_COLUMN = "customer_id"
TARGET = "churned"
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "age",
]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET_SERVICES = ["DSL", "Fiber", "no_internet"]
PAYMENT_METHODS = ["electronic_check", "credit_card", "bank_transfer", "mailed_check"]


def generate_customers(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create the synthetic dataset. The churn rules below are the 'ground truth'."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET_SERVICES, n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENT_METHODS, n, p=[0.35, 0.25, 0.20, 0.20])
    age = rng.integers(18, 80, n).astype(float)
    num_products = rng.integers(1, 6, n)

    # Long contracts tend to belong to customers who have been around longer.
    tenure = np.where(
        contract == "month-to-month",
        np.minimum(rng.exponential(14, n), 72),
        np.where(contract == "one-year", rng.uniform(6, 72, n), rng.uniform(12, 72, n)),
    ).astype(int)

    base_price = pd.Series(internet).map({"DSL": 25.0, "Fiber": 50.0, "no_internet": 0.0}).to_numpy()
    monthly = np.clip(20 + base_price + 8 * num_products + rng.normal(0, 5, n), 18, 150).round(2)
    total = (monthly * tenure * rng.normal(1.0, 0.03, n)).round(2)

    # Unhappy customers call support often. Calls accumulate over the customer's lifetime,
    # so the raw count mixes "unhappiness" with "how long they've been here".
    complaint_rate = rng.gamma(shape=4.0, scale=0.1, size=n)  # calls per month, hidden
    support_calls = rng.poisson(complaint_rate * (tenure + 1))

    # ---- Ground-truth churn rules (log-odds) ----
    logit = (
        -2.6
        + 1.4 * (contract == "month-to-month")
        - 1.0 * (contract == "two-year")
        - 0.03 * tenure
        + 0.015 * (monthly - 70)
        + 0.5 * (internet == "Fiber")
        + 0.6 * (payment == "electronic_check")
        + 4.0 * (complaint_rate - 0.4)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(size=n) < churn_probability).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_products": num_products,
            "support_calls": support_calls,
            "age": age,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            TARGET: churned,
        }
    )

    # Real data has holes. Brand-new customers have not been billed yet, and some
    # fields were simply never collected.
    df.loc[df["tenure_months"] == 0, "total_charges"] = np.nan
    df.loc[rng.uniform(size=n) < 0.06, "age"] = np.nan
    df.loc[rng.uniform(size=n) < 0.03, "payment_method"] = np.nan
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset, generating it on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_customers().to_csv(path, index=False)
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2):
    """Split into train / validation / test (default 60 / 20 / 20).

    - train:      the model learns from this
    - validation: we compare models, tune settings, and pick a threshold on this
    - test:       touched ONCE at the very end, to estimate real-world performance

    `stratify` keeps the churn rate the same in every split, which matters when one
    class is rare.
    """
    X, y = df[FEATURES], df[TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_size / (1 - test_size),
        stratify=y_rest,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Turn raw JSON-like records into the exact frame shape the model was trained on.

    Used by the API so that serving data looks like training data (same columns,
    same order, missing values as NaN rather than None).
    """
    df = pd.DataFrame.from_records(records).reindex(columns=FEATURES)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df
