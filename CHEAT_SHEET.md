# ML cheat sheet (1 page) — starter version, rewrite it in your own words

## Core vocabulary
| Term | Meaning |
|---|---|
| Feature (X) | Input column known *at prediction time* |
| Label / target (y) | What we predict |
| Train / Val / Test | Fit / compare + tune / final honest estimate (use once) |
| Cross-validation | k train/score rounds on train data → mean ± std; more reliable than one split |
| Baseline | Simplest model (Dummy, logistic regression). Anything fancier must beat it |
| Overfitting | Great on train, worse on val → simplify, regularise, add data |
| Leakage | Info from val/test or from the future gets into training → fake-good scores |

## Regression vs classification
| | Regression (number) | Classification (category) |
|---|---|---|
| Example | monthly bill $ | churn yes/no |
| Models | LinearRegression, RandomForestRegressor | LogisticRegression, RandomForestClassifier, HistGradientBoosting |
| Metrics | **MAE** (avg error, in units), **RMSE** (punishes big misses), **R²** (variance explained) | **Precision** TP/(TP+FP), **Recall** TP/(TP+FN), **F1**, **ROC-AUC**, **PR-AUC** (rare class) |

Choose the metric by asking **what a mistake costs**. Missed churner (FN) costly → recall. Limited outreach budget → precision / PR-AUC.

## Pipeline template
```python
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
cross_validate(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True), scoring=["average_precision"])
pipe.fit(X_train, y_train); joblib.dump(pipe, "model.joblib")
```
Everything that *learns from data* (impute, scale, encode, select, resample) goes **inside** the pipeline.

## Imbalanced data
1. Right metric (never accuracy alone) → 2. tune **threshold** on validation → 3. `class_weight="balanced"` → 4. resample (train fold only).
Class weighting changes *decisions*, not ranking (ROC-AUC barely moves).

## Feature engineering
Ratios/rates, flags/thresholds, interactions, date parts, aggregates. Helps linear models most. Keep it inside the pipeline (no train/serve skew). Check that the gain is larger than CV noise (std).

## Explainability
- **Coefficients** (scaled inputs): sign = direction, exp(coef) = odds ratio. Correlated features share or flip weight.
- **Permutation importance** on validation: model-agnostic, "how much worse when this column is shuffled".
- **Error analysis**: confusion matrix → errors per segment → most confident mistakes → decide next fix.
- Importance ≠ causation.

## Serving (FastAPI)
Load the model once at startup · validate input with Pydantic · save the threshold + version with the model · same `records → DataFrame` code for training and serving · test `/predict` with TestClient.

## Mistakes checklist
- [ ] Reported a training score as performance
- [ ] Used accuracy on imbalanced data / no dummy baseline
- [ ] Fit scaler/selector/imputer before splitting
- [ ] A feature only known *after* the outcome (target leakage)
- [ ] Tuned on test / looked at test many times
- [ ] Random split on time-ordered data · same entity in train and test
- [ ] Strings like `"None"`/`"NA"` silently read as missing by pandas
- [ ] Score "too good to be true" and not investigated
