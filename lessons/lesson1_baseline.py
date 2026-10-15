"""WEEK 1 - ML concepts: features, labels, train / validation / test.
Build: a baseline classifier.

Run:  python -m lessons.lesson1_baseline

Key ideas
---------
* Supervised learning = learn a function  f(features) -> label  from examples.
* Features (X): columns we know at prediction time. Label (y): what we want to predict.
* We split the data so we can measure how the model does on customers it has NOT seen:
    train      -> fit the model
    validation -> compare models / tune settings (you may look at it many times)
    test       -> final, honest estimate (look at it ONCE, at the end)
* A baseline is the simplest thing that could work. Every later model must beat it,
  otherwise the extra complexity isn't paying for itself.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from mlcourse.data import FEATURES, NUMERIC_FEATURES, TARGET, load_data, split_data
from mlcourse.evaluate import classification_metrics, header, show_table


def main() -> None:
    header("1. Look at the data")
    df = load_data()
    print(df.head().to_string())
    print(f"\nRows: {len(df)}   Feature columns: {len(FEATURES)}   Label: '{TARGET}'")
    print(f"Churn rate: {df[TARGET].mean():.1%}  <- the classes are imbalanced")
    print("\nMissing values per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string())

    header("2. Split into train / validation / test (60 / 20 / 20, stratified)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    for name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"{name:<11} {len(y):>5} rows   churn rate {y.mean():.1%}")

    header("3. Baseline #0: always predict the most common class")
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    results = {"dummy (always 'stays')": classification_metrics(y_val, dummy.predict_proba(X_val)[:, 1])}
    print("High accuracy, zero recall: it never catches a single churner.")
    print("This is why accuracy alone is misleading on imbalanced data.")

    header("4. Baseline #1: logistic regression on numeric columns only")
    # Models need numbers with no gaps. Fill missing values with the TRAINING median.
    # Computing the median on all data would leak information from val/test into training.
    train_medians = X_train[NUMERIC_FEATURES].median()
    Xtr = X_train[NUMERIC_FEATURES].fillna(train_medians)
    Xva = X_val[NUMERIC_FEATURES].fillna(train_medians)

    model = LogisticRegression(max_iter=2000).fit(Xtr, y_train)
    results["logreg (numeric only)"] = classification_metrics(y_val, model.predict_proba(Xva)[:, 1])
    show_table(results, "Validation metrics")

    print("\nTrain vs validation (a big gap would mean overfitting):")
    show_table(
        {
            "train": classification_metrics(y_train, model.predict_proba(Xtr)[:, 1]),
            "validation": results["logreg (numeric only)"],
        }
    )

    header("5. Final check on the TEST set (do this once, at the very end)")
    Xte = X_test[NUMERIC_FEATURES].fillna(train_medians)
    show_table({"logreg (numeric only)": classification_metrics(y_test, model.predict_proba(Xte)[:, 1])})

    print(
        """
EXERCISES
  1. Remove `stratify` from split_data. How much does the churn rate vary between splits?
  2. Drop 'support_calls' from the features. Which metric drops most?
  3. Why is roc_auc of the dummy model exactly 0.5?
  4. In your own words: why must test data be used only once?
"""
    )


if __name__ == "__main__":
    main()
