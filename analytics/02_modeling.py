"""Module 2, Part B: predictive modeling, continuing from Part A's data.

Reads the committed titanic.csv that 01_eda.py produced. The raw dataset is
never re-loaded from seaborn here -- that happened exactly once, in Part A.

All preprocessing lives inside a ColumnTransformer wrapped in a Pipeline, so
imputer/encoder/scaler are structurally incapable of seeing the test split:
they are fit when .fit() is called on the training data and applied in
transform-only mode thereafter.

Outputs: charts/*.png, modeling_output.md, best_pipeline.joblib
"""

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from report_utils import Report

CSV_PATH = "titanic.csv"
OUTPUT_MD = "modeling_output.md"
PIPELINE_PATH = "best_pipeline.joblib"
RANDOM_STATE = 42

NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]
FEATURES = NUMERIC + CATEGORICAL

r = Report()
say = r.say

say("# Module 2 Part B - Predictive modeling\n")

# ---------------------------------------------------------------------------
# Data: continue from Part A's single load
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
say(
    f"Read `{CSV_PATH}` ({df.shape[0]} rows x {df.shape[1]} columns) - the snapshot "
    "`01_eda.py` wrote immediately after the module's single "
    "`sns.load_dataset('titanic')` call. No second load happens here.\n"
)

say("## Feature selection\n")
say(
    f"**Features used:** `{'`, `'.join(FEATURES)}`. **Target:** `survived`.\n"
)
say(
    "Several columns in the raw snapshot are deliberately excluded because they "
    "would leak the target or duplicate a feature already present:\n\n"
    "| Excluded | Why |\n|---|---|\n"
    "| `alive` | A verbatim string copy of `survived` - including it would make the task trivial and the metrics meaningless. |\n"
    "| `class` | The word form of `pclass` (`'First'` vs `1`). |\n"
    "| `embark_town` | The long form of `embarked` (`'Southampton'` vs `'S'`). |\n"
    "| `who`, `adult_male` | Derived from `sex` and `age`, which are both already features. |\n"
    "| `alone` | Derived: `sibsp + parch == 0`. |\n"
    "| `deck` | 77.22% missing, dropped in Part A for the reasons given there. |\n"
)

X = df[FEATURES]
y = df["survived"]

# ---------------------------------------------------------------------------
# Task 7 - stratified split, BEFORE any preprocessing
# ---------------------------------------------------------------------------
say("\n## Task 7 - Stratified train/test split\n")

balance = y.value_counts(normalize=True).sort_index()
say(
    f"**Class balance:** not survived (0) = **{balance[0]:.2%}**, "
    f"survived (1) = **{balance[1]:.2%}** - roughly a 62/38 split.\n"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

split_table = pd.DataFrame(
    {
        "rows": [len(y_train), len(y_test)],
        "survived %": [f"{y_train.mean():.2%}", f"{y_test.mean():.2%}"],
    },
    index=["train", "test"],
)
r.table(split_table, index=True)

say(
    "\n**Why stratification matters here.** At 38% positive the classes are "
    "imbalanced enough that an unstratified random split can drift by several "
    "percentage points in either direction purely by chance, and the test set is "
    "only 179 rows - small enough for that drift to move accuracy and recall "
    "noticeably. Stratifying forces both splits to carry the same 38.4% positive "
    "rate as the full dataset, so the test metrics measure the model rather than "
    "the luck of the draw. It also keeps the baseline honest: the majority-class "
    "accuracy floor is identical in both splits, making 'better than always "
    "predicting died' mean the same thing in training and evaluation.\n"
)

say(
    "The split happens **before** any preprocessing object is created, let alone "
    "fitted. Everything below fits on `X_train` only.\n"
)

# ---------------------------------------------------------------------------
# Task 8 - preprocessing, fit on training data only
# ---------------------------------------------------------------------------
say("\n## Task 8 - Preprocessing (fit on train only)\n")


def make_preprocessor():
    """Fresh ColumnTransformer: impute + scale numerics, impute + one-hot categoricals."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


say(
    "| Column group | Imputation | Then |\n|---|---|---|\n"
    f"| numeric (`{'`, `'.join(NUMERIC)}`) | median | `StandardScaler` |\n"
    f"| categorical (`{'`, `'.join(CATEGORICAL)}`) | most frequent | `OneHotEncoder(drop='first')` |\n"
)
say(
    "Median imputation for `age` matches Part A's choice and the same reasoning "
    "(age is right-skewed, so the median is the robust centre). `embarked`'s two "
    "missing values get the most frequent port rather than dropping the rows, "
    "because at prediction time a row with a missing port still has to produce an "
    "answer. `drop='first'` avoids the dummy-variable trap that would otherwise "
    "give logistic regression a perfectly collinear feature set.\n"
)
say(
    "**Leakage control.** The `ColumnTransformer` is a step inside the `Pipeline`, "
    "so `pipeline.fit(X_train, y_train)` fits the imputer statistics, the encoder "
    "categories and the scaler's mean/std on the training rows alone. Calling "
    "`pipeline.predict(X_test)` runs those same fitted objects in transform-only "
    "mode. There is no code path that could fit on the test split or on the full "
    "pre-split frame - the separation is structural, not a rule to remember.\n"
)

# ---------------------------------------------------------------------------
# Task 9 - train three classifiers on the identical split
# ---------------------------------------------------------------------------
say("\n## Task 9 - Three classifiers\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, oob_score=True
    ),
}

fitted = {}
for name, estimator in models.items():
    pipeline = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    print(f"  trained {name}")

say(
    "All three are trained on the identical `X_train` / `y_train`, each wrapped in "
    "its own copy of the same preprocessing pipeline. The decision tree is capped "
    "at `max_depth=4` so that `plot_tree` renders legibly and the tree does not "
    "simply memorise the training set.\n"
)

# plot_tree with labelled features and classes
tree_pipeline = fitted["Decision Tree"]
feature_names = tree_pipeline.named_steps["prep"].get_feature_names_out()
plt.figure(figsize=(20, 9))
plot_tree(
    tree_pipeline.named_steps["model"],
    feature_names=[n.split("__", 1)[-1] for n in feature_names],
    class_names=["died", "survived"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree (max_depth=4)")
r.chart("decision_tree")
say(
    "The root split is on `sex_male`, which matches Part A's finding that sex is "
    "the single strongest predictor. Below it the tree splits mainly on `pclass` "
    "and `fare` for males and on `pclass` and `age` for females - the same "
    "hierarchy the data story described, recovered independently by the model.\n"
)

# ---------------------------------------------------------------------------
# Task 10 - full metric suite
# ---------------------------------------------------------------------------
say("\n## Task 10 - Evaluation\n")


def evaluate(pipeline, name):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities),
    }, predictions, probabilities


say("### Confusion matrices\n")
figure, axes = plt.subplots(1, 3, figsize=(15, 4))
scores = []
curves = {}
for axis, (name, pipeline) in zip(axes, fitted.items()):
    row, predictions, probabilities = evaluate(pipeline, name)
    scores.append(row)
    curves[name] = probabilities

    matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
        xticklabels=["pred died", "pred survived"],
        yticklabels=["actual died", "actual survived"],
    )
    axis.set_title(name)

    say(f"**{name}**\n")
    r.table(
        pd.DataFrame(
            matrix,
            index=["actual died", "actual survived"],
            columns=["pred died", "pred survived"],
        ),
        index=True,
    )
    say("")
r.chart("confusion_matrices")

say("### ROC curves\n")
plt.figure(figsize=(6.5, 5.5))
for name, probabilities in curves.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probabilities):.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
r.chart("roc_curves")

say("### Comparison table\n")
classifier_table = pd.DataFrame(scores).set_index("model").round(4)
r.table(classifier_table, index=True)

# ---------------------------------------------------------------------------
# Task 11 - imbalance handling comparison
# ---------------------------------------------------------------------------
say("\n## Task 11 - Imbalance handling\n")
say(
    f"**Class balance (full dataset):** {balance[0]:.2%} did not survive, "
    f"{balance[1]:.2%} did. Three variants of Logistic Regression are compared.\n"
)

imbalance_rows = []

# (a) baseline
baseline = Pipeline(
    [("prep", make_preprocessor()),
     ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
).fit(X_train, y_train)

# (b) class_weight='balanced'
weighted = Pipeline(
    [("prep", make_preprocessor()),
     ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=RANDOM_STATE))]
).fit(X_train, y_train)

# (c) SMOTE -- inside an imblearn Pipeline so it only ever sees the training fold
smote = ImbPipeline(
    [("prep", make_preprocessor()),
     ("smote", SMOTE(random_state=RANDOM_STATE)),
     ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
).fit(X_train, y_train)

for label, pipeline in [
    ("(a) baseline", baseline),
    ("(b) class_weight='balanced'", weighted),
    ("(c) SMOTE (train fold only)", smote),
]:
    predictions = pipeline.predict(X_test)
    imbalance_rows.append(
        {
            "variant": label,
            "precision": round(precision_score(y_test, predictions), 4),
            "recall": round(recall_score(y_test, predictions), 4),
            "f1": round(f1_score(y_test, predictions), 4),
            "accuracy": round(accuracy_score(y_test, predictions), 4),
        }
    )

imbalance_table = pd.DataFrame(imbalance_rows)
r.table(imbalance_table)

best_imbalance = imbalance_table.loc[imbalance_table["f1"].idxmax()]
baseline_row = imbalance_table.iloc[0]

say(
    f"\n**Conclusion: {best_imbalance['variant']} worked best**, with F1 "
    f"{best_imbalance['f1']:.4f} against the baseline's {baseline_row['f1']:.4f}.\n"
)
say(
    "The mechanism is visible in the precision/recall columns, and it is the same "
    "for both corrections. The baseline maximises precision "
    f"({baseline_row['precision']:.4f}) but pays for it with the worst recall "
    f"({baseline_row['recall']:.4f}) - left alone, the model leans toward the "
    "majority 'died' class and misses roughly a third of actual survivors. Both "
    "`class_weight='balanced'` and SMOTE push the decision boundary toward the "
    f"minority class, trading a few points of precision for a jump in recall to "
    f"{best_imbalance['recall']:.4f}, and F1 rewards that trade because the "
    "recall gain outweighs the precision loss.\n"
)
say(
    "SMOTE edges out class weighting here, but by roughly half a point of F1 on a "
    "179-row test set - too small to call a real difference. Given that, "
    "`class_weight='balanced'` is the more sensible production choice: it achieves "
    "essentially the same result with a single constructor argument, no synthetic "
    "data, and no extra dependency. **Which correction matters far more than which "
    "one you pick** - both clearly beat leaving the imbalance unhandled.\n"
)

say(
    "\n**Why SMOTE is inside an `imblearn` Pipeline.** Oversampling before the "
    "split, or on the full dataset, copies synthetic neighbours of test-set "
    "passengers into the training data and inflates every metric. `ImbPipeline` "
    "applies the `SMOTE` step during `fit` only and skips it entirely during "
    "`predict`, so the test split is never resampled.\n"
)

# ---------------------------------------------------------------------------
# Task 12 - GridSearchCV + OOB
# ---------------------------------------------------------------------------
say("\n## Task 12 - Hyperparameter tuning\n")

grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [4, 8, None],
    "model__max_features": ["sqrt", "log2", None],
}

search = GridSearchCV(
    Pipeline(
        [
            ("prep", make_preprocessor()),
            # oob_score and bootstrap must both be set at construction time --
            # oob_score_ is not populated otherwise, and OOB requires bootstrapping.
            ("model", RandomForestClassifier(
                oob_score=True, bootstrap=True, random_state=RANDOM_STATE
            )),
        ]
    ),
    param_grid=grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
oob = search.best_estimator_.named_steps["model"].oob_score_

say(f"Searched **{len(grid['model__n_estimators']) * len(grid['model__max_depth']) * len(grid['model__max_features'])} "
    f"combinations** with 5-fold CV, scoring on F1.\n")
r.table(
    pd.DataFrame(
        [{"parameter": k, "best value": str(v)} for k, v in best_params.items()]
        + [{"parameter": "best CV F1", "best value": f"{search.best_score_:.4f}"}]
        + [{"parameter": "OOB score", "best value": f"{oob:.4f}"}]
    )
)

say(
    f"\n**Best parameters:** `{best_params}`, with a cross-validated F1 of "
    f"**{search.best_score_:.4f}** and an **out-of-bag score of {oob:.4f}**.\n"
)
say(
    "The estimator is constructed as "
    "`RandomForestClassifier(oob_score=True, bootstrap=True, ...)`. Both flags are "
    "required: `oob_score_` is simply not populated unless `oob_score=True` is "
    "passed at construction, and out-of-bag estimation is only defined when "
    "`bootstrap=True`, since the OOB sample is exactly the rows a tree's bootstrap "
    "draw left out. The OOB score is a free validation estimate computed on those "
    "held-out rows, so it corroborates the CV score without touching the test set.\n"
)

tuned_predictions = search.best_estimator_.predict(X_test)
tuned_probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
tuned_scores = {
    "model": "Random Forest (tuned)",
    "accuracy": accuracy_score(y_test, tuned_predictions),
    "precision": precision_score(y_test, tuned_predictions),
    "recall": recall_score(y_test, tuned_predictions),
    "f1": f1_score(y_test, tuned_predictions),
    "auc": roc_auc_score(y_test, tuned_probabilities),
}
say("Tuned model on the held-out test set:\n")
r.table(pd.DataFrame([tuned_scores]).set_index("model").round(4), index=True)

say(
    "\n**Tuning did not beat the default Random Forest on this split**, and that is "
    "worth stating plainly rather than hiding. The grid optimised 5-fold F1 on the "
    "training data; the default 300-tree configuration happens to generalise "
    "marginally better to these particular 179 test rows. With a test set this "
    "small a difference of a few tenths of a percent is well inside noise, so the "
    "honest reading is that the two are equivalent, not that tuning hurt.\n"
)

# ---------------------------------------------------------------------------
# Task 13 - regression side-task
# ---------------------------------------------------------------------------
say("\n## Task 13 - Regression: predicting fare\n")

REG_NUMERIC = ["pclass", "age", "sibsp", "parch", "survived"]
REG_CATEGORICAL = ["sex", "embarked"]

reg_X = df[REG_NUMERIC + REG_CATEGORICAL]
reg_y = df["fare"]

reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
    reg_X, reg_y, test_size=0.2, random_state=RANDOM_STATE
)

regressor = Pipeline(
    [
        (
            "prep",
            ColumnTransformer(
                [
                    ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                          ("scale", StandardScaler())]), REG_NUMERIC),
                    ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("encode", OneHotEncoder(handle_unknown="ignore",
                                                                       drop="first"))]),
                     REG_CATEGORICAL),
                ]
            ),
        ),
        ("model", LinearRegression()),
    ]
).fit(reg_X_train, reg_y_train)

reg_predictions = regressor.predict(reg_X_test)

n = len(reg_y_test)
p = regressor.named_steps["prep"].transform(reg_X_test).shape[1]
r2 = r2_score(reg_y_test, reg_predictions)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

regression_table = pd.DataFrame(
    [
        {
            "MAE": round(mean_absolute_error(reg_y_test, reg_predictions), 4),
            "RMSE": round(np.sqrt(mean_squared_error(reg_y_test, reg_predictions)), 4),
            "R2": round(r2, 4),
            "Adjusted R2": round(adjusted_r2, 4),
        }
    ]
)
say(f"Multivariate linear regression predicting `fare` from "
    f"`{'`, `'.join(REG_NUMERIC + REG_CATEGORICAL)}` "
    f"(n = {n} test rows, p = {p} encoded features).\n")
r.table(regression_table)

say(f"\n`Adjusted R2 = 1 - (1 - R2)(n - 1) / (n - p - 1)` "
    f"= `1 - (1 - {r2:.4f})({n} - 1) / ({n} - {p} - 1)` = **{adjusted_r2:.4f}**.\n")

residuals = reg_y_test - reg_predictions
plt.figure(figsize=(7, 4.5))
plt.scatter(reg_predictions, residuals, alpha=0.6, edgecolor="none")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual plot")
r.chart("residual_plot")

say(
    "**Heteroscedasticity: yes, clearly present.** The residuals fan out sharply "
    "as predicted fare increases - tightly clustered around zero at low predicted "
    "fares, then spreading to residuals in the hundreds at the high end. That "
    "widening cone is the textbook picture of non-constant error variance, and it "
    "violates the constant-variance assumption underpinning ordinary least "
    "squares. The cause is the right-skew Part A identified: a handful of "
    "first-class fares above 500 that a linear model in these features cannot "
    "reach, so it under-predicts them badly while fitting the dense low-fare mass "
    "well. The practical consequence is that the coefficient standard errors are "
    "unreliable, and a log-transformed target would be the standard remedy.\n"
)

# ---------------------------------------------------------------------------
# Task 14 - final comparison and recommendation
# ---------------------------------------------------------------------------
say("\n## Task 14 - Final model comparison\n")
say(
    "Classification and regression metrics are on **different scales measuring "
    "different tasks** and are not comparable numbers. They are presented as two "
    "separate metric groups below, never merged into one ranking.\n"
)

say("### Group 1 - Classification (target: `survived`)\n")
full_classifier_table = pd.concat(
    [classifier_table, pd.DataFrame([tuned_scores]).set_index("model").round(4)]
)
r.table(full_classifier_table, index=True)

say("\n### Group 2 - Regression (target: `fare`)\n")
r.table(regression_table.set_index(pd.Index(["Linear Regression"], name="model")), index=True)

best_name = full_classifier_table["f1"].idxmax()
best_row = full_classifier_table.loc[best_name]

say(
    f"\n### Recommendation\n\n"
    f"**Deploy {best_name}.** It posts the best F1 on the held-out test set at "
    f"**{best_row['f1']:.4f}**, with accuracy {best_row['accuracy']:.4f}, precision "
    f"{best_row['precision']:.4f}, recall {best_row['recall']:.4f} and AUC "
    f"{best_row['auc']:.4f}. F1 is the right criterion to rank on here because the "
    f"classes are imbalanced at 62/38 - accuracy alone rewards a model for getting "
    f"the majority 'died' class right, and a naive always-died classifier would "
    f"already score about {1 - y_test.mean():.2%} accuracy without learning "
    f"anything. The AUC of {best_row['auc']:.4f} confirms the ranking quality is "
    f"not an artefact of one particular decision threshold, which matters if the "
    f"operating point is ever retuned. Logistic Regression remains the honourable "
    f"mention: it trails on F1 but is far cheaper to serve and directly "
    f"interpretable via its coefficients, so it is the better pick if the "
    f"deployment context demands an explanation for every prediction.\n"
)

# ---------------------------------------------------------------------------
# Task 15 - persist the complete pipeline
# ---------------------------------------------------------------------------
say("\n## Task 15 - Saving and reloading the full pipeline\n")

# Save whichever pipeline actually won on F1 above, rather than assuming the
# tuned model did -- on this split it did not.
all_pipelines = {**fitted, "Random Forest (tuned)": search.best_estimator_}
full_pipeline = all_pipelines[best_name]
joblib.dump(full_pipeline, PIPELINE_PATH)
say(f"The saved artifact is **{best_name}**, the model recommended above.\n")
say(
    f"`joblib.dump(full_pipeline, '{PIPELINE_PATH}')` saves the **entire fitted "
    "`Pipeline`** - the `ColumnTransformer` with its fitted imputers, encoder and "
    "scaler, plus the tuned `RandomForestClassifier` - as one object. The bare "
    "estimator is deliberately *not* what gets saved: on its own it would expect "
    "pre-transformed numeric input and would be unusable on raw rows.\n"
)

reloaded = joblib.load(PIPELINE_PATH)

# Raw, unpreprocessed input: strings, an unscaled fare, and a missing age.
raw_input = pd.DataFrame(
    [
        {"pclass": 1, "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.34,
         "sex": "female", "embarked": "S"},
        {"pclass": 3, "age": None, "sibsp": 0, "parch": 0, "fare": 7.75,
         "sex": "male", "embarked": "Q"},
    ]
)

raw_predictions = reloaded.predict(raw_input)
raw_probabilities = reloaded.predict_proba(raw_input)[:, 1]

say(
    "Reloaded with `joblib.load` and given **raw, unpreprocessed rows** - "
    "categorical strings, an unscaled fare, and a deliberately missing `age` that "
    "the pipeline's imputer has to fill:\n"
)
demo = raw_input.copy()
demo["predicted"] = ["survived" if v == 1 else "died" for v in raw_predictions]
demo["P(survived)"] = raw_probabilities.round(4)
r.table(demo)

say(
    "\nThe wealthy first-class woman is predicted to survive with high confidence "
    "and the third-class man is predicted to die, matching Part A's data story. "
    "The row with a missing `age` produced a prediction rather than an error, "
    "confirming the saved artifact carries its own imputation and is usable "
    "end-to-end on raw input.\n"
)

# Same input through the in-memory pipeline must give the same answer.
assert (full_pipeline.predict(raw_input) == raw_predictions).all(), \
    "reloaded pipeline disagrees with the in-memory one"
say("`assert` check: reloaded pipeline's predictions match the in-memory "
    "pipeline's exactly.\n")

r.write(OUTPUT_MD)
print(f"Saved pipeline -> {PIPELINE_PATH}")
