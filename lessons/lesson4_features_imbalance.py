"""WEEK 4 - Feature engineering + class imbalance.
Build: improve the baseline and DOCUMENT WHY (fill in docs/EXPERIMENT_LOG.md).

Run:  python -m lessons.lesson4_features_imbalance

Key ideas
---------
Feature engineering = turning domain knowledge into columns the model can use.
  * Ratios / rates      (calls per month instead of total calls)
  * Thresholds / flags  (is_new_customer)
  * Interactions, date parts, aggregates, text lengths, ...
  Linear models gain the most: they can't invent ratios or steps by themselves.

Class imbalance (only ~18% churn). Options, from simplest to most invasive:
  1. Use the right metric (PR-AUC, recall, F1 - never plain accuracy).
  2. Move the decision THRESHOLD (0.5 is arbitrary; choose it on validation data).
  3. class_weight="balanced" (errors on the rare class cost more during training).
  4. Resampling (over/under-sampling, SMOTE) - only ever on the TRAINING fold.

Experiment discipline: change ONE thing at a time, compare with cross-validation on the
training set, and keep the test set for the final number only.
"""

from sklearn.model_selection import StratifiedKFold, cross_validate

from mlcourse.data import RANDOM_STATE, load_data, split_data
from mlcourse.evaluate import best_f1_threshold, classification_metrics, header, show_table
from mlcourse.pipeline import build_pipeline

EXPERIMENTS = {
    # name: (model, engineer_features, balanced)
    "E0 logreg baseline": ("logreg", False, False),
    "E1 + class_weight balanced": ("logreg", False, True),
    "E2 + engineered features": ("logreg", True, False),
    "E3 + features + balanced": ("logreg", True, True),
    "E4 gboost + features + balanced": ("gradient_boosting", True, True),
}


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_data())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    header("1. Experiments: 5-fold cross-validation on TRAIN (threshold 0.5)")
    rows = {}
    for name, (model, features, balanced) in EXPERIMENTS.items():
        scores = cross_validate(
            build_pipeline(model, features, balanced), X_train, y_train, cv=cv,
            scoring=["average_precision", "roc_auc", "precision", "recall", "f1"],
        )
        rows[name] = {
            "pr_auc": scores["test_average_precision"].mean(),
            **{m: scores[f"test_{m}"].mean() for m in ["roc_auc", "precision", "recall", "f1"]},
        }
    table = show_table(rows)

    print(
        """
How to read this:
  * class_weight (E0 -> E1): PR-AUC/ROC-AUC barely move - ranking quality is the same -
    but recall jumps because the model now predicts 'churn' far more often at 0.5.
    Balancing mostly SHIFTS the threshold; it does not make the model smarter.
  * features (E0 -> E2): PR-AUC and ROC-AUC go UP. That's real new signal:
    calls_per_month separates unhappy customers from merely long-standing ones.
  * gradient boosting (E4) is not automatically better. With good features and a
    mostly-linear problem, the simple model wins - and it's easier to explain."""
    )

    best = table["pr_auc"].idxmax()
    model, features, balanced = EXPERIMENTS[best]
    header(f"2. Best by PR-AUC: '{best}'. Tune the threshold on VALIDATION")
    pipe = build_pipeline(model, features, balanced).fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    print(f"Best-F1 threshold on validation: {threshold:.3f}  (default was 0.5)")
    show_table(
        {
            "threshold 0.5": classification_metrics(y_val, val_proba, 0.5),
            f"threshold {threshold:.2f}": classification_metrics(y_val, val_proba, threshold),
        },
        "Validation",
    )

    header("3. Final, one-time comparison on TEST")
    baseline = build_pipeline("logreg").fit(X_train, y_train)
    show_table(
        {
            "baseline (E0 @ 0.5)": classification_metrics(y_test, baseline.predict_proba(X_test)[:, 1]),
            f"improved ({best.split()[0]} @ {threshold:.2f})": classification_metrics(
                y_test, pipe.predict_proba(X_test)[:, 1], threshold
            ),
        }
    )
    print(
        """
Accuracy may go DOWN while the model got better: we now flag more customers, so
some extra false alarms appear - in exchange we catch many more real churners.
Small differences on test (+/- 0.01-0.02) are noise with only 1200 rows; trust the
cross-validated comparison for choosing, and the test set for the final estimate.

YOUR TURN: copy the numbers above into docs/EXPERIMENT_LOG.md and write, for each
experiment, what you changed, what happened, and WHY you think it happened.

EXERCISES
  1. Add a feature in mlcourse/features.py (e.g. total_charges / monthly_charges, or
     age buckets). Run again. Did PR-AUC move by more than its CV noise (~0.01-0.02)?
  2. Try oversampling churners in the training data instead of class_weight. Make sure
     you oversample ONLY training rows - what goes wrong if you oversample before splitting?
  3. Pick the threshold that gives recall >= 0.80 instead of best F1. What precision do you get?
"""
    )


if __name__ == "__main__":
    main()
