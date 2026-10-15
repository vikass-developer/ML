"""WEEK 2 - Regression vs classification + metrics.
Build: compare 2 models with metrics.

Run:  python -m lessons.lesson2_regression_vs_classification

Key ideas
---------
* Regression     -> predict a NUMBER     (e.g. monthly bill in $)
  Classification -> predict a CATEGORY   (e.g. churn yes/no)
* Regression metrics
    MAE  = average |error|            -> "off by $X on average", easy to explain
    RMSE = sqrt(mean(error^2))        -> punishes big misses more than MAE
    R^2  = 1 - SSE/SST                -> share of variance explained (1 = perfect, 0 = predicting the mean)
* Classification metrics (positive class = churned)
    precision = TP / (TP + FP)        -> of the customers we flagged, how many really churned?
    recall    = TP / (TP + FN)        -> of the customers who churned, how many did we catch?
    F1        = harmonic mean of precision and recall
    ROC-AUC   = how well the model RANKS churners above non-churners (0.5 = random)
    PR-AUC    = like ROC-AUC but focused on the positive class; better when it's rare
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mlcourse.data import FEATURES, RANDOM_STATE, TARGET, load_data, split_data
from mlcourse.evaluate import classification_metrics, header, show_table


def one_hot(train: pd.DataFrame, other: pd.DataFrame):
    """Manual preprocessing: fill gaps + one-hot encode. Note how fiddly this is:
    `other` must be forced to have exactly the training columns. Week 3 replaces
    all of this with a Pipeline."""
    medians = train.median(numeric_only=True)
    train_enc = pd.get_dummies(train.fillna(medians), dtype=float)
    other_enc = pd.get_dummies(other.fillna(medians), dtype=float)
    other_enc = other_enc.reindex(columns=train_enc.columns, fill_value=0.0)
    return train_enc, other_enc


def regression_part(df: pd.DataFrame) -> None:
    header("PART A - Regression: predict monthly_charges ($)")
    features = ["internet_service", "num_products", "contract", "age", "tenure_months"]
    X_train, X_val, y_train, y_val = train_test_split(
        df[features], df["monthly_charges"], test_size=0.25, random_state=RANDOM_STATE
    )
    X_train, X_val = one_hot(X_train, X_val)

    models = {
        "baseline: predict the mean": DummyRegressor(strategy="mean"),
        "linear regression": LinearRegression(),
        "random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE),
    }
    rows = {}
    for name, model in models.items():
        pred = model.fit(X_train, y_train).predict(X_val)
        rows[name] = {
            "MAE": mean_absolute_error(y_val, pred),
            "RMSE": np.sqrt(mean_squared_error(y_val, pred)),
            "R2": r2_score(y_val, pred),
        }
    show_table(rows, "Validation metrics (lower MAE/RMSE is better, higher R2 is better)")
    print(
        "\nThe price is (almost) a linear formula of internet type + products, so linear"
        "\nregression is as good as the forest and far simpler/more explainable."
        "\nThe remaining error (~$4) is the random noise we added: no model can predict that."
    )


def classification_part(df: pd.DataFrame) -> None:
    header("PART B - Classification: predict churn (0/1)")
    X_train, X_val, _, y_train, y_val, _ = split_data(df)
    X_train, X_val = one_hot(X_train[FEATURES], X_val[FEATURES])

    models = {
        "logistic regression": LogisticRegression(max_iter=5000),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE),
    }
    rows = {}
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            print(
                f"\n[note] {name} hit a ConvergenceWarning: features are on very different scales"
                "\n       (total_charges ~ thousands, is_fiber ~ 0/1). Week 3 adds StandardScaler to fix it."
            )
        proba = model.predict_proba(X_val)[:, 1]
        rows[name] = classification_metrics(y_val, proba)

        tn, fp, fn, tp = confusion_matrix(y_val, (proba >= 0.5).astype(int)).ravel()
        print(f"\n{name} - confusion matrix @ threshold 0.5")
        print(f"                 predicted stay   predicted churn")
        print(f"  actual stay    TN = {tn:<11} FP = {fp}")
        print(f"  actual churn   FN = {fn:<11} TP = {tp}")

    show_table(rows, "Validation metrics")
    print(
        """
How to choose the metric = ask what a mistake COSTS:
  * FN (missed churner)  -> we lose a customer worth months of revenue
  * FP (false alarm)     -> we give a discount to someone who'd have stayed
  If FN is more expensive, prioritise RECALL. If the retention team can only call
  50 people a week, prioritise PRECISION among the top 50 (ranking -> PR-AUC).
  Recall at 0.5 is low for both models: Week 4 fixes that."""
    )


def main() -> None:
    df = load_data()
    regression_part(df)
    classification_part(df)
    print(
        """
EXERCISES
  1. Compute MAE and RMSE by hand with numpy for the linear model and check they match.
  2. Change the threshold from 0.5 to 0.3 in classification_metrics. What happens to
     precision and recall? Why does ROC-AUC not change?
  3. Which model would you ship for churn, and which ONE metric would you report to a
     business stakeholder? Write two sentences justifying it.
"""
    )


if __name__ == "__main__":
    main()
