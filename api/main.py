"""WEEK 6 - Mini-project: predictive API with FastAPI.

1. Train + save a model:   python -m mlcourse.train --engineer-features --balanced
2. Start the API:          uvicorn api.main:app --reload
3. Try it in the browser:  http://127.0.0.1:8000/docs   (interactive Swagger UI)

Design notes
------------
* The model is loaded ONCE at startup, not on every request.
* Pydantic validates input BEFORE it reaches the model: wrong types, negative tenure,
  or an unknown contract type return a clear 422 error instead of a silent bad prediction.
* The saved bundle includes the pipeline AND the decision threshold, so the API uses
  exactly what was chosen during training.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from mlcourse.data import records_to_frame
from mlcourse.train import DEFAULT_MODEL_PATH


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=600, description="Months as a customer")
    monthly_charges: float = Field(gt=0, le=1000)
    total_charges: float | None = Field(default=None, ge=0, description="Empty for brand-new customers")
    num_products: int = Field(ge=1, le=20)
    support_calls: int = Field(ge=0, description="Total support calls over the customer's lifetime")
    age: float | None = Field(default=None, ge=18, le=120)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["DSL", "Fiber", "no_internet"]
    payment_method: Literal["electronic_check", "credit_card", "bank_transfer", "mailed_check"] | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure_months": 3,
                    "monthly_charges": 95.5,
                    "total_charges": 286.5,
                    "num_products": 3,
                    "support_calls": 4,
                    "age": 34,
                    "contract": "month-to-month",
                    "internet_service": "Fiber",
                    "payment_method": "electronic_check",
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    model_version: str


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"No model at {path}. Train one first: python -m mlcourse.train --engineer-features --balanced"
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    yield


app = FastAPI(title="Churn prediction API", version="1.0", lifespan=lifespan)


def predict_customers(request: Request, customers: list[Customer]) -> list[Prediction]:
    bundle = request.app.state.bundle
    X = records_to_frame([c.model_dump() for c in customers])
    probabilities = bundle["pipeline"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]
    version = bundle["metadata"]["model_version"]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            will_churn=bool(p >= threshold),
            threshold=round(threshold, 4),
            model_version=version,
        )
        for p in probabilities
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    return request.app.state.bundle["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    return predict_customers(request, [customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], request: Request) -> list[Prediction]:
    if not 1 <= len(customers) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 customers")
    return predict_customers(request, customers)
