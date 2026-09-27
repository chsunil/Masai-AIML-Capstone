# Module 2 — Analytics Pipeline

One cohesive pipeline: the dataset is loaded from the network exactly once, and
every later step continues from that single load.

## Run it

```bash
cd analytics
python 01_eda.py        # sns.load_dataset -> titanic.csv, charts/, eda_output.md   (needs network once)
python 02_modeling.py   # titanic.csv -> charts/, modeling_output.md, best_pipeline.joblib
```

`01_eda.py` is the only place `sns.load_dataset('titanic')` is called. It writes
`titanic.csv` immediately after loading, and `02_modeling.py` reads that
committed file — so the module is fully gradeable offline via
`pd.read_csv("titanic.csv")` even with no network at grading time.

Scripts rather than notebooks; the brief states these are equally acceptable,
and it keeps the module consistent with Module 1's script-plus-committed-output
pattern. Every printed result is captured in the two generated Markdown reports.

## Where the written interpretations live

| Report | Covers |
|---|---|
| **[`eda_output.md`](eda_output.md)** | Tasks 1–6: profiling, missing-value strategy, univariate/bivariate analysis, correlation heatmap, the 5-chart data story, standardization check |
| **[`modeling_output.md`](modeling_output.md)** | Tasks 7–15: split, preprocessing, three classifiers, metrics, imbalance comparison, tuning, regression, final table and recommendation |

Charts are in [`charts/`](charts/) and embedded in both reports.

## Headline results

**Missing values** — `deck` 77.22% (column dropped), `age` 19.87% (median
imputed), `embarked`/`embark_town` 0.22% (2 rows dropped). Each strategy cites
the measured percentage against the brief's threshold rule.

**`fare` is right-skewed** — mode 8.05 < median 14.45 < mean 32.10, with 114 IQR
outliers forming the long right tail.

**Two strongest correlations** — `pclass`/`fare` (−0.548) and `sibsp`/`parch`
(0.415). Both are structural rather than insights; the strongest *target*
correlation is `survived`/`pclass` at −0.336, only fourth overall.

**Classification** (179-row stratified test set):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| Decision Tree | 0.7877 | 0.8444 | 0.5507 | 0.6667 | 0.8210 |
| **Random Forest** | **0.8045** | 0.7742 | 0.6957 | **0.7328** | 0.8267 |
| Random Forest (tuned) | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8333 |

**Regression** (target `fare`) — MAE 20.90, RMSE 30.53, R² 0.3975,
Adjusted R² 0.3692. The residual plot shows clear heteroscedasticity.

**Imbalance** — SMOTE scored best (F1 0.7606) over `class_weight='balanced'`
(0.7552) and baseline (0.7244), but the gap between the two corrections is
inside noise; both clearly beat leaving the imbalance unhandled.

**Tuning** — GridSearchCV over 27 combinations found
`max_depth=8, max_features=None, n_estimators=100` with CV F1 0.7586 and
**OOB score 0.8258**. It did not beat the default Random Forest on the test
set, which the report states rather than glossing over.

## Design decisions

### Leakage control is structural, not procedural

All preprocessing lives in a `ColumnTransformer` inside a `Pipeline`. Fitting
happens only when `.fit(X_train, y_train)` is called; `.predict(X_test)` runs
the same fitted objects in transform-only mode. There is no code path that
could fit on test data, so the guarantee does not depend on remembering a rule.

SMOTE uses `imblearn.pipeline.Pipeline` rather than scikit-learn's — the
imblearn version applies resampling during `fit` only and skips it during
`predict`. A plain sklearn Pipeline would resample the test split too.

### Excluded columns

`alive` is a verbatim copy of `survived` and would make the task trivial.
`class`, `embark_town`, `who`, `adult_male` and `alone` are all derived from
columns already in the feature set. `deck` was dropped in Part A.

### OOB requires two flags

`RandomForestClassifier(oob_score=True, bootstrap=True)` — `oob_score_` is not
populated unless `oob_score=True` is passed at *construction*, and out-of-bag
estimation is only defined when bootstrapping is on.

### The saved artifact is the recommended model

`best_pipeline.joblib` holds the complete fitted `Pipeline` — the
`ColumnTransformer` with its imputers, encoder and scaler, plus the final
estimator — selected by whichever model actually won on F1, not by assuming the
tuned one did. It is verified by reloading and predicting on raw rows including
one with a missing `age`, with an `assert` confirming the reloaded pipeline
agrees with the in-memory one.
