"""Reusable training script (Week 3 deliverable).

    python -m mlcourse.train                                   # logistic regression
    python -m mlcourse.train --model gradient_boosting --engineer-features --balanced

Steps: load -> split -> fit on train -> choose threshold on validation
       -> evaluate ONCE on test -> save a bundle the API can load.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from mlcourse.data import FEATURES, PROJECT_ROOT, load_data, split_data
from mlcourse.evaluate import best_f1_threshold, classification_metrics
from mlcourse.pipeline import MODEL_NAMES, build_pipeline

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def train_and_save(
    model_name: str = "logreg",
    engineer_features: bool = False,
    balanced: bool = False,
    out_path: Path = DEFAULT_MODEL_PATH,
) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_data())

    pipeline = build_pipeline(model_name, engineer_features, balanced)
    pipeline.fit(X_train, y_train)

    threshold = best_f1_threshold(y_val, pipeline.predict_proba(X_val)[:, 1])
    test_metrics = classification_metrics(y_test, pipeline.predict_proba(X_test)[:, 1], threshold)

    # Save the model together with everything needed to use and audit it.
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "features": FEATURES,
        "metadata": {
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "model_name": model_name,
            "engineer_features": engineer_features,
            "balanced": balanced,
            "sklearn_version": sklearn.__version__,
            "n_train": len(X_train),
            "threshold": round(threshold, 4),
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return bundle["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the churn model.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="logreg")
    parser.add_argument("--engineer-features", action="store_true")
    parser.add_argument("--balanced", action="store_true", help="use class_weight='balanced'")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    metadata = train_and_save(args.model, args.engineer_features, args.balanced, args.out)
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved model bundle to {args.out}")


if __name__ == "__main__":
    main()
