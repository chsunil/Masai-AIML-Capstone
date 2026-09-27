"""Module 2, Part A: profile the Titanic dataset, clean it, and tell its story.

This is the ONE AND ONLY place the raw dataset is loaded from the network or
seaborn's cache. Immediately after loading it is written to titanic.csv, which
is committed; 02_modeling.py reads that file and never calls load_dataset again.

Outputs: titanic.csv, charts/*.png, eda_output.md
"""

import matplotlib

matplotlib.use("Agg")  # no display in this environment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from report_utils import Report

CSV_PATH = "titanic.csv"
CHART_DIR = "charts"
OUTPUT_MD = "eda_output.md"

report_obj = Report()
say = report_obj.say
chart = report_obj.chart
block = report_obj.block
report = report_obj.lines


# ---------------------------------------------------------------------------
# Task 1 - load once, snapshot, profile
# ---------------------------------------------------------------------------
say("# Module 2 Part A - EDA and data story\n")
say("## Task 1 - Loading and profiling\n")

df = sns.load_dataset("titanic")
df.to_csv(CSV_PATH, index=False)
say(
    f"Loaded via `sns.load_dataset('titanic')` and immediately snapshotted to "
    f"`{CSV_PATH}` ({len(df)} rows). This is the only raw load in the module; "
    f"`02_modeling.py` reads that committed CSV.\n"
)

say(f"`df.shape` -> **{df.shape}**\n")

info_buffer = []
df.info(buf=type("W", (), {"write": lambda self, s: info_buffer.append(s)})())
block("".join(info_buffer), "df.info()")
block(df.describe(), "df.describe()")

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
say("**Missing values, as a percentage of all 891 rows:**\n")
report.append(
    missing_pct.rename("missing_%").to_frame().to_markdown() + "\n"
)
print(missing_pct.to_string())

# ---------------------------------------------------------------------------
# Task 2 - missing-value handling under the threshold rule
# ---------------------------------------------------------------------------
say("\n## Task 2 - Missing-value handling\n")
say(
    "Threshold rule from the brief: **under 5% missing -> drop those rows; "
    "5%-30% -> impute**; above that, decide explicitly and justify.\n"
)

deck_pct = missing_pct["deck"]
age_pct = missing_pct["age"]
embarked_pct = missing_pct["embarked"]

say(
    f"| Column | Measured missing | Band | Strategy applied |\n"
    f"|---|---|---|---|\n"
    f"| `deck` | **{deck_pct}%** | above 30% | **Drop the column** |\n"
    f"| `age` | **{age_pct}%** | 5-30% | **Impute** with the median |\n"
    f"| `embarked` | **{embarked_pct}%** | under 5% | **Drop those rows** |\n"
    f"| `embark_town` | **{missing_pct['embark_town']}%** | under 5% | Same 2 rows as `embarked` |\n"
)

say(
    f"**`deck` ({deck_pct}%) - dropped rather than encoded.** At {deck_pct}% missing "
    "only ~203 of 891 passengers have a recorded deck, and that recording is not "
    "random: deck was transcribed mainly from surviving first-class cabin records. "
    "Encoding `'Missing'` as its own category would therefore create a level that is "
    "largely a proxy for *not first class*, duplicating information `pclass` already "
    "carries cleanly and completely. Imputing a value for 77% of rows would be "
    "fabrication, so the column is dropped.\n"
)
say(
    f"**`age` ({age_pct}%) - median-imputed.** This sits squarely in the 5-30% "
    "impute band. The median is used rather than the mean because age is "
    "right-skewed, so the median is the more robust centre.\n"
)
say(
    f"**`embarked` / `embark_town` ({embarked_pct}%) - rows dropped.** Two rows, "
    "comfortably under the 5% threshold. Dropping them costs 0.22% of the data.\n"
)

clean = df.drop(columns=["deck"]).copy()
clean = clean.dropna(subset=["embarked"]).copy()
clean["age"] = clean["age"].fillna(clean["age"].median())

say(f"Cleaned shape: **{clean.shape}** (from {df.shape}). Remaining missing values: "
    f"**{int(clean.isna().sum().sum())}**\n")

# ---------------------------------------------------------------------------
# Task 3 - univariate analysis
# ---------------------------------------------------------------------------
say("\n## Task 3 - Univariate analysis\n")

for column in ["age", "fare"]:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(clean[column], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"{column}: distribution")
    sns.boxplot(x=clean[column], ax=axes[1], color="#DD8452")
    axes[1].set_title(f"{column}: box plot")
    chart(f"univariate_{column}")


def iqr_outliers(series):
    """Count points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < low) | (series > high)).sum()), low, high


say("**IQR outlier counts** (outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`):\n")
rows = []
for column in ["age", "fare"]:
    count, low, high = iqr_outliers(clean[column])
    rows.append(
        {
            "column": column,
            "lower fence": round(low, 2),
            "upper fence": round(high, 2),
            "outliers": count,
            "% of rows": round(100 * count / len(clean), 2),
        }
    )
outlier_table = pd.DataFrame(rows)
report.append(outlier_table.to_markdown(index=False) + "\n")
print(outlier_table.to_string(index=False))

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]

say(
    f"\n**`fare` central tendency:** mean = **{fare_mean:.2f}**, "
    f"median = **{fare_median:.2f}**, mode = **{fare_mode:.2f}**.\n"
)
say(
    f"Mode ({fare_mode:.2f}) < median ({fare_median:.2f}) < mean ({fare_mean:.2f}). "
    "That ordering - mean pulled furthest to the right, above the median, which in "
    "turn sits above the mode - is the textbook signature of a **right-skewed "
    "(positively skewed) distribution**. The long right tail comes from a small "
    "number of very expensive first-class fares dragging the mean upward while the "
    f"bulk of passengers cluster near the mode; the {rows[1]['outliers']} IQR "
    "outliers in `fare` are that same tail.\n"
)

# ---------------------------------------------------------------------------
# Task 4 - bivariate analysis
# ---------------------------------------------------------------------------
say("\n## Task 4 - Bivariate analysis\n")
say("Survival rates computed with boolean masking (`&` / `|` combinations).\n")

is_female = clean["sex"] == "female"
survived_mask = clean["survived"] == 1

say("**(a) By sex**\n")
sex_rows = [
    {"sex": "female", "survival rate": round(survived_mask[is_female].mean(), 4),
     "n": int(is_female.sum())},
    {"sex": "male", "survival rate": round(survived_mask[~is_female].mean(), 4),
     "n": int((~is_female).sum())},
]
report.append(pd.DataFrame(sex_rows).to_markdown(index=False) + "\n")
print(pd.DataFrame(sex_rows).to_string(index=False))

say("\n**(b) By passenger class**\n")
class_rows = []
for pclass in [1, 2, 3]:
    mask = clean["pclass"] == pclass
    class_rows.append(
        {"pclass": pclass, "survival rate": round(survived_mask[mask].mean(), 4),
         "n": int(mask.sum())}
    )
report.append(pd.DataFrame(class_rows).to_markdown(index=False) + "\n")
print(pd.DataFrame(class_rows).to_string(index=False))

say("\n**(c) By sex AND class** (combined masks)\n")
combo_rows = []
for pclass in [1, 2, 3]:
    for label, sex_mask in [("female", is_female), ("male", ~is_female)]:
        mask = sex_mask & (clean["pclass"] == pclass)
        combo_rows.append(
            {"sex": label, "pclass": pclass,
             "survival rate": round(survived_mask[mask].mean(), 4),
             "n": int(mask.sum())}
        )
combo_table = pd.DataFrame(combo_rows)
report.append(combo_table.to_markdown(index=False) + "\n")
print(combo_table.to_string(index=False))

# correlation matrix on exactly the six specified columns
CORR_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[CORR_COLUMNS].corr()

say(
    "\n**Correlation matrix** on exactly the six specified columns "
    "(`adult_male` and `alone` are excluded: they are derived flags computable "
    "from `sex`/`age` and from `sibsp`+`parch`, not independent measurements).\n"
)
report.append(corr.round(3).to_markdown() + "\n")
print(corr.round(3).to_string())

plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix (6 specified columns)")
chart("correlation_heatmap")

# rank all off-diagonal pairs by absolute correlation
pairs = [
    (a, b, corr.loc[a, b])
    for i, a in enumerate(CORR_COLUMNS)
    for b in CORR_COLUMNS[i + 1:]
]
pairs.sort(key=lambda p: abs(p[2]), reverse=True)
top_two = pairs[:2]

say("**All off-diagonal pairs ranked by |correlation|:**\n")
ranked = pd.DataFrame(
    [{"pair": f"{a} - {b}", "correlation": round(v, 4), "abs": round(abs(v), 4)}
     for a, b, v in pairs]
)
report.append(ranked.to_markdown(index=False) + "\n")
print(ranked.to_string(index=False))

(a1, b1, v1), (a2, b2, v2) = top_two
say(
    f"\n**Strongest pair: `{a1}` and `{b1}` ({v1:.3f}).** Fare and class are two "
    "measurements of the same underlying thing - what a passenger paid and the "
    "tier that bought them. The sign is negative only because `pclass` is coded "
    "with 1 as the *best* class, so a lower class number goes with a higher fare. "
    "It is the strongest relationship in the matrix precisely because it is close "
    "to tautological rather than a discovered insight.\n"
)
say(
    f"**Second strongest: `{a2}` and `{b2}` ({v2:.3f}).** Both columns count family "
    "members aboard - siblings/spouses and parents/children respectively - so they "
    "rise and fall together: a passenger travelling as part of a family group tends "
    "to have non-zero values in both. Like the first pair this is structural rather "
    "than a discovered insight, and it is a warning for the modeling stage: the two "
    "features are partly redundant, which is exactly why the derived `alone` flag "
    "(computable from `sibsp + parch == 0`) was excluded from this matrix.\n"
)
say(
    "Worth noting what is **not** in the top two. The strongest correlation "
    f"involving the target is `survived`-`pclass` at "
    f"{corr.loc['survived', 'pclass']:.3f}, only fourth overall, followed by "
    f"`survived`-`fare` at {corr.loc['survived', 'fare']:.3f}. Every correlation "
    "with `survived` is modest, and that is expected rather than discouraging: the "
    "dominant survival driver is `sex`, a categorical column that cannot appear in "
    "a numeric correlation matrix at all. Task 4(a) above shows that effect "
    "plainly, and it is invisible here.\n"
)

# ---------------------------------------------------------------------------
# Task 5 - multivariate data story
# ---------------------------------------------------------------------------
say("\n## Task 5 - Multivariate data story: who survived, and why\n")

# Chart 1
plt.figure(figsize=(7, 4.5))
sns.barplot(data=clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by class and sex")
plt.ylabel("survival rate")
chart("story_1_class_sex")
say(
    "**Chart 1 - Survival rate by class and sex.** This is the central fact of the "
    "dataset: within every single class, women survived at far higher rates than "
    "men. First-class women survived at roughly 97%, while third-class men survived "
    "at under 14%. Class modulates the effect but never reverses it - even "
    "third-class women outperformed first-class men. Any model that fails to use "
    "`sex` is discarding the strongest available signal.\n"
)

# Chart 2
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=clean, x="pclass", y="fare", hue="survived")
plt.ylim(0, 300)
plt.title("Fare distribution by class and survival (y clipped at 300)")
chart("story_2_fare_class")
say(
    "**Chart 2 - Fare by class and survival.** Within first class, survivors paid "
    "noticeably more than non-survivors, and the same gap is faintly visible in "
    "second and third class. This suggests fare carries information *beyond* the "
    "class label alone - plausibly cabin location on the upper decks, closer to the "
    "boat deck. The y-axis is clipped at 300 because a handful of extreme fares "
    "(the IQR outliers found in Task 3) otherwise flatten the whole plot.\n"
)

# Chart 3
plt.figure(figsize=(7.5, 4.5))
sns.scatterplot(data=clean, x="age", y="fare", hue="survived", alpha=0.6,
                palette={0: "#C44E52", 1: "#55A868"})
plt.ylim(0, 300)
plt.title("Age vs fare, coloured by survival (y clipped at 300)")
chart("story_3_age_fare")
say(
    "**Chart 3 - Age against fare.** Survivors (green) dominate the upper band of "
    "the chart at essentially every age, confirming that fare separates outcomes "
    "more sharply than age does. Along the bottom - the cheap-ticket band where "
    "most passengers sit - the two colours are thoroughly mixed, which is why `fare` "
    "alone will not be a reliable predictor. The visible cluster of survivors among "
    "very young children at low fares is the 'children first' effect surviving even "
    "in third class.\n"
)

# Chart 4
plt.figure(figsize=(7, 4.5))
pivot = clean.pivot_table(index="who", columns="pclass", values="survived", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
plt.title("Survival rate by passenger type and class")
chart("story_4_who_class")
say(
    "**Chart 4 - Survival rate by passenger type and class.** Splitting into "
    "child / woman / man makes the 'women and children first' protocol explicit: "
    "children and women in first and second class survived at near-ceiling rates, "
    "while men are the dark band across all three classes. The sharpest drop in the "
    "entire grid is third-class children, who fared dramatically worse than children "
    "in the upper classes - the protocol was applied, but access to the boat deck "
    "was not equal.\n"
)

# Chart 5
plt.figure(figsize=(7, 4.5))
sns.barplot(data=clean, x="alone", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by travelling alone, split by sex")
plt.ylabel("survival rate")
chart("story_5_alone_sex")
say(
    "**Chart 5 - Travelling alone.** Passengers travelling with family survived at "
    "higher rates than those travelling alone, but the effect is strongly "
    "sex-dependent: it is large for women and nearly flat for men. This is most "
    "likely confounding rather than causation - women travelling with family were "
    "disproportionately in the upper classes, and family groups were prioritised "
    "for lifeboats. It justifies keeping `sibsp` and `parch` as model features "
    "while treating `alone` itself as redundant with them.\n"
)

say(
    "**The story in one paragraph.** Survival on the Titanic was determined first "
    "by sex, second by class, and third by age - in that order. Being female "
    "roughly quadrupled a passenger's chance of survival; being in first rather "
    "than third class roughly doubled it again; being a child helped, but only if "
    "you were not in third class. Fare adds a little information beyond class, and "
    "family structure adds a little beyond that, but both are largely downstream of "
    "the same socio-economic divide. A good classifier should therefore lean "
    "heavily on `sex` and `pclass`, with `age` and `fare` as refinements.\n"
)

# ---------------------------------------------------------------------------
# Task 6 - exploratory standardization check
# ---------------------------------------------------------------------------
say("\n## Task 6 - Exploratory standardization check\n")
say(
    "`z = (x - mean) / std` applied to `age` and `fare` on the **full cleaned "
    "DataFrame**. This is an EDA sanity check only - it deliberately does **not** "
    "feed into the modeling pipeline, which fits its own `StandardScaler` on the "
    "training split alone to avoid leakage.\n"
)

standardized = pd.DataFrame(
    {f"{c}_z": (clean[c] - clean[c].mean()) / clean[c].std() for c in ["age", "fare"]}
)

before_after = pd.DataFrame(
    {
        "before: mean": [clean["age"].mean(), clean["fare"].mean()],
        "before: std": [clean["age"].std(), clean["fare"].std()],
        "after: mean": [standardized["age_z"].mean(), standardized["fare_z"].mean()],
        "after: std": [standardized["age_z"].std(), standardized["fare_z"].std()],
    },
    index=["age", "fare"],
).round(6)
report.append(before_after.to_markdown() + "\n")
print(before_after.to_string())

say(
    "\nBoth transformed columns land on mean ~0 and standard deviation ~1 "
    "(the tiny residuals are floating-point error), confirming the formula was "
    "applied correctly.\n"
)

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
for axis, column in zip(axes, ["age", "fare"]):
    sns.kdeplot(clean[column], ax=axis, label="before", fill=True)
    sns.kdeplot(standardized[f"{column}_z"], ax=axis, label="after (z-score)", fill=True)
    axis.set_title(f"{column}: before vs after standardization")
    axis.legend()
chart("standardization_before_after")
say(
    "Note the shape of each distribution is unchanged - standardization shifts and "
    "rescales the axis but does not make a skewed variable symmetric. `fare` is "
    "just as right-skewed after the transform as before it.\n"
)

# ---------------------------------------------------------------------------
with open(OUTPUT_MD, "w", encoding="utf-8") as handle:
    handle.write("\n".join(report))
print(f"\nWrote report -> {OUTPUT_MD}")
print(f"Wrote dataset snapshot -> {CSV_PATH}")
