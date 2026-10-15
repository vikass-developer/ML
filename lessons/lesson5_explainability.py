"""WEEK 5 - Model explainability basics.
Build: feature importance + error analysis.

Run:  python -m lessons.lesson5_explainability

Key ideas
---------
* GLOBAL explanation: which features matter for the model overall?
    - Linear model coefficients (on standardised inputs): sign = direction,
      size = strength. exp(coef) = odds ratio per 1 std increase.
    - Permutation importance: shuffle one column on VALIDATION data and measure how
      much the score drops. Works for ANY model. Drop ~0 -> the model doesn't need it.
* LOCAL explanation: why did the model score THIS customer high?
    - For a linear model: contribution_i = coef_i * scaled_value_i (they add up to the logit).
    - For other models, libraries like SHAP generalise this idea.
* ERROR ANALYSIS: where is the model wrong? Look at the confusion matrix, error rates
  per segment, and the most confident mistakes. This tells you what to fix next
  (more data for a segment, a new feature, a label problem...).
* Importance != causation. "Fiber users churn more" does not mean fiber causes churn.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from mlcourse.data import load_data, split_data
from mlcourse.evaluate import best_f1_threshold, header
from mlcourse.pipeline import build_pipeline


def main() -> None:
    X_train, X_val, _, y_train, y_val, _ = split_data(load_data())
    pipe = build_pipeline("logreg", engineer_features=True, balanced=True).fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, proba)

    header("1. Global: logistic regression coefficients (standardised inputs)")
    names = pipe[:-1].get_feature_names_out()
    coefs = pd.DataFrame({"coef": pipe.named_steps["model"].coef_[0]}, index=names)
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    print(coefs.round(3).to_string())
    print(
        "\n+ coef -> pushes toward churn, - coef -> toward staying."
        "\nCompare with the ground truth in mlcourse/data.py: month-to-month, fiber,"
        "\nelectronic check, high calls_per_month raise churn; tenure and two-year lower it."
        "\nCaution 1: correlated features (tenure/total_charges, support_calls/calls_per_month)"
        "\nsplit or even flip their weights - read them as a group, not one by one."
        "\nCaution 2: with one-hot columns, compare categories to EACH OTHER, not to zero:"
        "\nFiber minus DSL is what tells you fiber users churn more."
    )

    header("2. Global: permutation importance on VALIDATION (model-agnostic)")
    result = permutation_importance(
        pipe, X_val, y_val, scoring="average_precision", n_repeats=10, random_state=0
    )
    perm = pd.DataFrame(
        {"pr_auc_drop": result.importances_mean, "std": result.importances_std}, index=X_val.columns
    ).sort_values("pr_auc_drop", ascending=False)
    print(perm.round(4).to_string())
    print(
        "\nThis is computed on RAW columns: shuffling support_calls also breaks calls_per_month,"
        "\nso its importance includes the engineered feature. age ~0 -> the model ignores it"
        "\n(correct: age has no effect in the ground truth)."
    )

    header("3. Local: why was this customer flagged?")
    i = int(np.argmax(proba))
    model_inputs = pipe[:-1].transform(X_val.iloc[[i]])[0]
    contrib = pd.Series(model_inputs * pipe.named_steps["model"].coef_[0], index=names)
    print(X_val.iloc[i].to_string())
    print(f"\nchurn probability: {proba[i]:.3f}   actually churned: {bool(y_val.iloc[i])}")
    print("\nTop contributions to the log-odds (intercept "
          f"{pipe.named_steps['model'].intercept_[0]:+.2f}):")
    print(contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6).round(3).to_string())

    header(f"4. Error analysis on VALIDATION (threshold {threshold:.2f})")
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    errors = X_val.assign(
        actual=y_val.to_numpy(), proba=proba, pred=pred,
        tenure_bucket=pd.cut(X_val["tenure_months"], [-1, 5, 23, 47, 120], labels=["0-5", "6-23", "24-47", "48+"]),
    )
    errors["false_negative"] = (errors["actual"] == 1) & (errors["pred"] == 0)
    errors["false_positive"] = (errors["actual"] == 0) & (errors["pred"] == 1)
    for segment in ["contract", "tenure_bucket", "internet_service"]:
        seg = errors.groupby(segment, observed=True).agg(
            customers=("actual", "size"),
            churn_rate=("actual", "mean"),
            recall=("pred", lambda p: p[errors.loc[p.index, "actual"] == 1].mean()),
            false_pos_rate=("false_positive", "mean"),
        )
        print(f"\nBy {segment}:")
        print(seg.round(3).to_string())

    cols = ["tenure_months", "contract", "monthly_charges", "support_calls", "proba"]
    print("\nMost confident MISSED churners (false negatives):")
    print(errors[errors["false_negative"]].nsmallest(5, "proba")[cols].round(3).to_string())
    print("\nMost confident FALSE ALARMS (false positives):")
    print(errors[errors["false_positive"]].nlargest(5, "proba")[cols].round(3).to_string())
    print(
        """
Reading the error analysis:
  * Recall is lowest for long-contract / long-tenure customers: the model has learned
    "these people stay", so the few who do leave look like stayers.
  * Confident misses with ordinary-looking features are often IRREDUCIBLE: churn here
    has randomness (as in real life - someone moves house). No feature can fix that.
  * Many "false alarms" are new, expensive, month-to-month customers - genuinely at risk,
    they just haven't left (yet). For a retention campaign, contacting them is fine.

EXERCISES
  1. Run permutation importance with scoring="roc_auc". Does the ranking change?
  2. Repeat sections 1-2 with build_pipeline("random_forest", True, True) and compare the
     forest's built-in feature_importances_ with permutation importance. Which do you trust?
  3. Pick one confident false negative and write 2 sentences on what data could have caught it.
"""
    )


if __name__ == "__main__":
    main()
