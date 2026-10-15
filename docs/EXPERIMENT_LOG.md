# Experiment log — churn model (Week 4 deliverable)

**Goal:** catch more churners than the baseline without drowning the retention team in false alarms.
**Primary metric:** PR-AUC (ranking quality on a rare class). **Decision metric:** recall and precision at the chosen threshold.
**Method:** 5-fold stratified CV on the training split only. Change one thing per experiment.

| ID | Change | PR-AUC | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|----|--------|-------:|--------:|-----:|-----:|-----:|
| E0 | Logistic regression, raw features | 0.463 | 0.805 | 0.569 | 0.187 | 0.281 |
| E1 | + `class_weight="balanced"` | 0.464 | 0.805 | 0.350 | 0.796 | 0.486 |
| E2 | + engineered features | | | | | |
| E3 | + features + balanced | | | | | |
| E4 | Gradient boosting + features + balanced | | | | | |

> Fill in E2–E4 from `python -m lessons.lesson4_features_imbalance`.

---

### E0 — Baseline (worked example)
- **What:** logistic regression on the 9 raw columns, impute + scale + one-hot.
- **Result:** ROC-AUC 0.81 is decent, so the model *ranks* customers reasonably well, but recall at 0.5 is only ~19%.
- **Why:** with 18% churners, the model rarely becomes >50% confident anyone will churn. The 0.5 cut-off doesn't suit this problem.

### E1 — Class weighting (worked example)
- **What:** `class_weight="balanced"`, so each churner counts about 4.5× during training.
- **Result:** PR-AUC/ROC-AUC unchanged; recall 0.19 → 0.80; precision 0.57 → 0.35.
- **Why:** reweighting pushes all probabilities up, which works like lowering the threshold. The *ranking*, and so ROC-AUC/PR-AUC, barely changes. It is a decision-policy change, not new knowledge.
- **Keep?** Yes, if the business values recall. Threshold tuning on validation achieves something similar.

### E2 — Engineered features (your turn)
- **What:**
- **Result:**
- **Why:** (hint: compare `support_calls` vs `calls_per_month` for a 2-month and a 60-month customer)
- **Keep?**

### E3 — Features + balanced (your turn)

### E4 — Gradient boosting (your turn)
- **Why might a more powerful model do worse here?** (hint: how was churn generated in `mlcourse/data.py`?)

---

### Final decision
- **Chosen model:**
- **Threshold (chosen on validation):**
- **Test-set result (looked at once):**
- **Known weaknesses (from Week 5 error analysis):**
