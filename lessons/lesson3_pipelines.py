"""WEEK 3 - scikit-learn pipelines + preprocessing.
Build: a reusable training pipeline (see mlcourse/pipeline.py and mlcourse/train.py).

Run:  python -m lessons.lesson3_pipelines
Then: python -m mlcourse.train --help

Key ideas
---------
* Preprocessing has parameters learned from data (medians, scaling means, category
  lists). They must be learned on TRAINING data only and re-used everywhere else.
* A Pipeline bundles preprocessing + model into ONE estimator:
    pipe.fit(X_train, y_train)    # fits imputer, scaler, encoder, model - in order
    pipe.predict_proba(X_new)     # re-applies the SAME learned transforms
* ColumnTransformer = different preprocessing for different column types.
* Cross-validation (CV): split the training data into k folds, train on k-1, score on
  the remaining one, repeat k times. More reliable than a single validation split.
  Because the preprocessing is inside the pipeline, it is re-fit on every fold -> no leakage.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from mlcourse.data import RANDOM_STATE, load_data, split_data
from mlcourse.evaluate import classification_metrics, header, show_table
from mlcourse.pipeline import MODEL_NAMES, build_pipeline


def main() -> None:
    X_train, X_val, _, y_train, y_val, _ = split_data(load_data())

    header("1. What a pipeline looks like")
    pipe = build_pipeline("logreg")
    print(pipe)

    header("2. Fit once, use everywhere")
    pipe.fit(X_train, y_train)  # raw DataFrame in: NaNs and strings are handled inside
    feature_names = pipe[:-1].get_feature_names_out()
    print(f"{X_train.shape[1]} raw columns became {len(feature_names)} model inputs:")
    print(list(feature_names))
    print("\nLearned on TRAINING data only:")
    num = pipe.named_steps["preprocess"].named_transformers_["num"]
    medians = num.named_steps["impute"].statistics_
    print("  medians used for imputation:", {c: round(float(m), 1) for c, m in zip(num.feature_names_in_, medians)})
    cat = pipe.named_steps["preprocess"].named_transformers_["cat"]
    print("  categories seen by encoder: ", [list(c) for c in cat.named_steps["onehot"].categories_])

    show_table({"logreg pipeline": classification_metrics(y_val, pipe.predict_proba(X_val)[:, 1])}, "Validation metrics")

    header("3. Compare models fairly with 5-fold cross-validation (on train only)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name in MODEL_NAMES:
        scores = cross_validate(build_pipeline(name), X_train, y_train, cv=cv,
                                scoring=["roc_auc", "average_precision", "f1"])
        rows[name] = {
            "roc_auc": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "pr_auc": scores["test_average_precision"].mean(),
            "f1@0.5": scores["test_f1"].mean(),
            "fit_seconds": scores["fit_time"].mean(),
        }
    show_table(rows, "Mean over 5 folds (std shows how stable the score is)")
    print(
        "\nIf two models are within ~1 std of each other, treat them as equal and pick"
        "\nthe simpler / faster / more explainable one."
    )

    header("4. Unseen category? The pipeline doesn't crash")
    weird = X_val.head(2).copy()
    weird["contract"] = "three-year"  # never seen in training
    print(pd.DataFrame({"churn_probability": pipe.predict_proba(weird)[:, 1].round(3)}))

    print(
        """
NEXT: train + save a model you can reuse:
  python -m mlcourse.train --engineer-features --balanced

EXERCISES
  1. Replace StandardScaler with MinMaxScaler in mlcourse/pipeline.py. Does logreg change? Does the forest?
     (Why are tree models insensitive to scaling?)
  2. Change the numeric imputer to strategy="mean". Any difference? When would it matter?
  3. Add a '--cv' flag to mlcourse/train.py that prints 5-fold CV scores before training.
"""
    )


if __name__ == "__main__":
    main()
