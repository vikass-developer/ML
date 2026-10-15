"""API tests. Run:  python -m pytest -q

A small model is trained into a temp folder, so the tests never depend on (or
overwrite) the model you trained yourself.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from mlcourse.train import train_and_save

RISKY = {
    "tenure_months": 1, "monthly_charges": 110.0, "total_charges": 110.0, "num_products": 3,
    "support_calls": 4, "age": 30, "contract": "month-to-month", "internet_service": "Fiber",
    "payment_method": "electronic_check",
}
SAFE = {
    "tenure_months": 70, "monthly_charges": 45.0, "total_charges": 3150.0, "num_products": 2,
    "support_calls": 2, "age": 55, "contract": "two-year", "internet_service": "DSL",
    "payment_method": "bank_transfer",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("model") / "churn_model.joblib"
    train_and_save("logreg", engineer_features=True, balanced=True, out_path=model_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(model_path))
        with TestClient(app) as c:  # `with` runs the startup (model loading)
            yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_has_test_metrics(client):
    info = client.get("/model-info").json()
    assert info["test_metrics"]["roc_auc"] > 0.75


def test_risky_customer_scores_higher_than_safe(client):
    risky = client.post("/predict", json=RISKY).json()
    safe = client.post("/predict", json=SAFE).json()
    assert risky["churn_probability"] > safe["churn_probability"]
    assert risky["will_churn"] is True
    assert safe["will_churn"] is False


def test_missing_optional_fields_are_imputed(client):
    customer = {**RISKY, "total_charges": None, "age": None, "payment_method": None}
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


def test_invalid_input_is_rejected(client):
    assert client.post("/predict", json={**RISKY, "contract": "lifetime"}).status_code == 422
    assert client.post("/predict", json={**RISKY, "tenure_months": -3}).status_code == 422


def test_batch(client):
    response = client.post("/predict/batch", json=[RISKY, SAFE])
    assert response.status_code == 200
    assert len(response.json()) == 2
