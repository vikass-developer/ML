"""Reusable preprocessing + model pipelines (Week 3).

Pipeline = [optional feature engineering] -> [preprocessing] -> [model]

    raw DataFrame
        │
        ├── numeric columns ──> impute median ──> standard scale ─┐
        └── categorical cols ─> impute mode ───> one-hot encode ──┴─> model

Every step is `fit` on training data only, then `transform`ed on anything else.
The whole thing is ONE object: fit it, save it, load it in the API, call predict.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlcourse.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from mlcourse.features import ENGINEERED_FEATURES, ChurnFeatureEngineer

MODEL_NAMES = ["logreg", "random_forest", "gradient_boosting"]


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": a category never seen in training becomes all
            # zeros instead of crashing the API.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", categorical_steps, categorical),
        ]
    )


def make_model(name: str, balanced: bool = False):
    # class_weight="balanced" makes mistakes on the rare class (churners) cost more.
    class_weight = "balanced" if balanced else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight=class_weight,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def build_pipeline(
    model_name: str = "logreg",
    engineer_features: bool = False,
    balanced: bool = False,
) -> Pipeline:
    steps = []
    numeric = list(NUMERIC_FEATURES)
    if engineer_features:
        steps.append(("features", ChurnFeatureEngineer()))
        numeric += ENGINEERED_FEATURES
    steps.append(("preprocess", build_preprocessor(numeric, CATEGORICAL_FEATURES)))
    steps.append(("model", make_model(model_name, balanced)))
    return Pipeline(steps)
