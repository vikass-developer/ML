"""Feature engineering (Week 4).

A feature-engineering step written as a scikit-learn transformer can live INSIDE
the pipeline. That gives two guarantees:
  1. It is applied identically at training time and in the API (no train/serve skew).
  2. During cross-validation it only ever sees the training fold (no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURES = ["calls_per_month", "charge_per_product", "is_new_customer"]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-driven features to the raw customer frame.

    - calls_per_month:    support_calls / (tenure + 1). The raw call count grows with
                          tenure, so it mixes "unhappy" with "loyal". The rate isolates
                          unhappiness. A linear model cannot compute a ratio by itself.
    - charge_per_product: how much each product costs the customer (price sensitivity).
    - is_new_customer:    first 6 months are the riskiest; a step a linear model can't draw.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to learn here, but if a feature needed statistics (e.g. a mean),
        # they MUST be computed in fit() from training data only.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["calls_per_month"] = X["support_calls"] / (X["tenure_months"] + 1)
        X["charge_per_product"] = X["monthly_charges"] / X["num_products"].clip(lower=1)
        X["is_new_customer"] = (X["tenure_months"] < 6).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + ENGINEERED_FEATURES, dtype=object)
