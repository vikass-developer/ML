"""WEEK 7 - Review ML concepts + common mistakes.
Build: a 1-page cheat sheet (start from CHEAT_SHEET.md and make it yours).

Run:  python -m lessons.lesson7_common_mistakes

Each demo below reproduces a classic mistake, shows the misleading number it
produces, and then the correct way.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from mlcourse.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data, split_data
from mlcourse.evaluate import classification_metrics, header
from mlcourse.pipeline import build_pipeline, build_preprocessor


def mistake_1_scoring_on_training_data(X_train, X_val, y_train, y_val) -> None:
    header("MISTAKE 1: judging a model by its training score")
    tree = make_pipeline(
        build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        DecisionTreeClassifier(random_state=0),  # no depth limit -> memorises
    ).fit(X_train, y_train)
    print(f"train accuracy:      {tree.score(X_train, y_train):.3f}   <- looks perfect")
    print(f"validation accuracy: {tree.score(X_val, y_val):.3f}   <- reality (dummy gets ~0.81)")
    print("FIX: always report validation / CV scores. A large train-val gap = overfitting.")


def mistake_2_accuracy_on_imbalanced(X_train, X_val, y_train, y_val) -> None:
    header("MISTAKE 2: using accuracy when one class is rare")
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    m = classification_metrics(y_val, dummy.predict_proba(X_val)[:, 1])
    print(f"'always predict stay': accuracy {m['accuracy']:.3f}, recall {m['recall']:.3f}")
    print("FIX: use precision / recall / F1 / PR-AUC, and always compare against a dummy baseline.")


def mistake_3_preprocessing_leakage() -> None:
    header("MISTAKE 3: preprocessing / feature selection BEFORE splitting (leakage)")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5000))   # 5000 columns of pure noise
    y = rng.integers(0, 2, 200)        # random labels: true accuracy is 50%

    # WRONG: pick the 20 "best" columns using ALL rows, then cross-validate.
    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
    wrong = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()

    # RIGHT: selection inside the pipeline -> re-done on each training fold only.
    right = cross_val_score(
        make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), X, y, cv=5
    ).mean()
    print(f"selection before CV:   {wrong:.3f}   <- 'works' on random noise!")
    print(f"selection inside CV:   {right:.3f}   <- the truth: coin flip")
    print("FIX: every step that learns from data (scaling, imputing, selecting, resampling)")
    print("     goes inside a Pipeline, fit on training data only.")


def mistake_4_target_leakage(X_train, X_val, y_train, y_val) -> None:
    header("MISTAKE 4: target leakage - a feature that secretly contains the answer")
    rng = np.random.default_rng(1)

    def add_leaky_column(X, y):
        # e.g. "cancellation_fee_charged" - only exists AFTER a customer churned.
        X = X.copy()
        X["cancellation_fee_charged"] = (y.to_numpy() == 1) & (rng.uniform(size=len(y)) < 0.9)
        X["cancellation_fee_charged"] = X["cancellation_fee_charged"].astype(int)
        return X

    numeric = NUMERIC_FEATURES + ["cancellation_fee_charged"]
    leaky = make_pipeline(build_preprocessor(numeric, CATEGORICAL_FEATURES), LogisticRegression(max_iter=2000))
    leaky.fit(add_leaky_column(X_train, y_train), y_train)
    m = classification_metrics(y_val, leaky.predict_proba(add_leaky_column(X_val, y_val))[:, 1])
    print(f"with leaky column: roc_auc {m['roc_auc']:.3f}, recall {m['recall']:.3f}  <- too good to be true")

    honest = build_pipeline("logreg").fit(X_train, y_train)
    print(f"honest model:      roc_auc {classification_metrics(y_val, honest.predict_proba(X_val)[:, 1])['roc_auc']:.3f}")
    print("FIX: for every feature ask 'would I know this value at the moment I predict?'")
    print("     Suspiciously high scores are a bug until proven otherwise.")


def mistake_5_tuning_on_test() -> None:
    header("MISTAKE 5: tuning on the test set (no demo - just don't)")
    print(
        "If you try 50 models and keep the best TEST score, that score is optimistic: you\n"
        "selected for luck. Choose models/thresholds on validation or CV; look at test once.\n"
        "Other silent mistakes: random split on time-ordered data (use a time-based split),\n"
        "the same customer in both train and test (use GroupKFold), and string values that\n"
        "pandas silently reads as missing ('None', 'NA' - this project hit that bug!)."
    )


def main() -> None:
    X_train, X_val, _, y_train, y_val, _ = split_data(load_data())
    mistake_1_scoring_on_training_data(X_train, X_val, y_train, y_val)
    mistake_2_accuracy_on_imbalanced(X_train, X_val, y_train, y_val)
    mistake_3_preprocessing_leakage()
    mistake_4_target_leakage(X_train, X_val, y_train, y_val)
    mistake_5_tuning_on_test()
    print("\nNow open CHEAT_SHEET.md and rewrite it in your own words - that's the deliverable.")


if __name__ == "__main__":
    main()
