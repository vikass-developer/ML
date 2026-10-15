# Learn ML by building: customer churn, from baseline to API

One project, seven weeks. Each week you learn a concept by running a lesson, reading its
code, and doing its exercises. The project grows into a trained, explained model served
through a FastAPI endpoint.

**The problem:** a telecom company wants to know which customers are likely to leave
(churn) so the retention team can contact them first. The data is synthetic and generated
in `mlcourse/data.py`, so you know the true rules and can check whether the model found them.

## Setup (Windows)
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Run everything from this folder (the one with this README).

## Roadmap
| Week | Concept | Run / read | Deliverable |
|---|---|---|---|
| 1 | Features, labels, train/val/test | `python -m lessons.lesson1_baseline` | Baseline classifier |
| 2 | Regression vs classification + metrics | `python -m lessons.lesson2_regression_vs_classification` | Compare 2 models with metrics |
| 3 | scikit-learn pipelines + preprocessing | `python -m lessons.lesson3_pipelines` · read [mlcourse/pipeline.py](mlcourse/pipeline.py), [mlcourse/train.py](mlcourse/train.py) | Reusable training pipeline: `python -m mlcourse.train --help` |
| 4 | Feature engineering + imbalance | `python -m lessons.lesson4_features_imbalance` · read [mlcourse/features.py](mlcourse/features.py) | Improved model + [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) |
| 5 | Explainability basics | `python -m lessons.lesson5_explainability` | Feature importance + error analysis |
| 6 | Mini-project: predictive API | read [api/main.py](api/main.py) | Model served with FastAPI (below) |
| 7 | Review + common mistakes | `python -m lessons.lesson7_common_mistakes` | Your own version of [CHEAT_SHEET.md](CHEAT_SHEET.md) |

**How to study each week**
1. Read the docstring at the top of the lesson (the theory, in short).
2. Run the lesson and read the output. Predict each number before you look at it.
3. Read the code line by line, then do the EXERCISES printed at the end.
4. Break something on purpose (remove a step, change a parameter) and explain what happens.

## Week 6: run the API
```powershell
python -m mlcourse.train --engineer-features --balanced   # writes models/churn_model.joblib
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs, choose **POST /predict**, then **Try it out** and **Execute**.
Other endpoints: `GET /health`, `GET /model-info` (metrics + threshold), `POST /predict/batch`.

```powershell
python -m pytest -q        # API tests (trains a throwaway model in a temp folder)
```

## Project layout
```
mlcourse/            reusable code (what you'd keep in a real project)
  data.py            dataset generator, column lists, train/val/test split
  features.py        feature engineering as a sklearn transformer
  pipeline.py        preprocessing + model pipelines
  evaluate.py        metrics + threshold selection
  train.py           CLI: train -> pick threshold -> test -> save bundle
lessons/             one runnable script per week (learning material)
api/main.py          FastAPI service
tests/test_api.py    API tests
docs/EXPERIMENT_LOG.md   Week 4: document what you tried and why
CHEAT_SHEET.md       Week 7: one-page summary
```

## Results you should see (approximately)
| | Recall (churners caught) | Precision | PR-AUC |
|---|---|---|---|
| Week 1 baseline (numeric only, threshold 0.5) | ~5–10% | ~0.5 | ~0.40 |
| Week 4 model (features + balanced + tuned threshold) | ~70% | ~0.40 | ~0.50 |

## After this course
- Swap the synthetic data for a real dataset (e.g. the Kaggle "Telco Customer Churn" CSV); only `mlcourse/data.py` needs to change.
- Add hyperparameter search (`GridSearchCV` around the pipeline).
- Try SHAP for richer explanations; add MLflow to track experiments; Dockerise the API.
