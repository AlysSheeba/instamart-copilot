from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")

orders = pd.read_csv(DATA / "orders.csv")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

orders["status"] = (
    orders["status"].astype("string").str.strip().str.lower()
)

print("Part B — Customer Segments and Lapse Prediction")
print(f"Orders loaded: {len(orders):,}")
print(f"Customers loaded: {len(customers):,}")
print(f"Products loaded: {len(products):,}")
# ============================================================
# PART B — TASKS 1–3: RFM, K-MEANS, SEGMENT PROFILES AND PCA
# ============================================================

print("\nTASK 1: BUILDING THE RFM TABLE")

completed = orders[
    (orders["status"] == "completed")
    & orders["order_ts"].notna()
].copy()

completed = completed[completed["customer_id"].notna()]
completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", lambda x: x.sum(min_count=1)),
).reset_index()

rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].dropna()

rfm = rfm[rfm["recency"] >= 0].copy()

print(f"RFM customers: {len(rfm):,}")
print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

print("\nTASK 2: K-MEANS CLUSTERING")

rfm_features = ["recency", "frequency", "monetary"]
scaler_rfm = StandardScaler()
scaled_rfm = scaler_rfm.fit_transform(rfm[rfm_features])

cluster_results = []
cluster_models = {}

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(scaled_rfm)
    silhouette = silhouette_score(scaled_rfm, labels)

    cluster_results.append({
        "K": k,
        "inertia": model.inertia_,
        "silhouette": silhouette,
    })
    cluster_models[k] = model

cluster_table = pd.DataFrame(cluster_results)
print("\nK-means comparison:")
print(cluster_table.round(3).to_string(index=False))

# Choose K=4 as a practical balance of segment detail,
# clustering quality, and business interpretability.
FINAL_K = 4
final_kmeans = cluster_models[FINAL_K]
rfm["segment_id"] = final_kmeans.labels_

profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "size"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)

premium_id = profile["mean_monetary"].idxmax()

# Give each cluster a readable name based on its behaviour.
segment_names = {}

for segment_id, row in profile.iterrows():
    if segment_id == premium_id:
        segment_names[segment_id] = "Premium Customers"
    elif row["mean_frequency"] >= profile["mean_frequency"].median():
        segment_names[segment_id] = "Frequent Customers"
    elif row["mean_recency"] >= profile["mean_recency"].median():
        segment_names[segment_id] = "At-Risk Customers"
    else:
        segment_names[segment_id] = "Developing Customers"

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

print("\nSegment profiles:")
print(profile.to_string())
print("\nSegment names:")
for segment_id, name in segment_names.items():
    print(f"Cluster {segment_id}: {name}")
print(f"Premium cluster ID: {premium_id}")

print(
    "\nInterpretation: K=8 will generally have lower inertia than K=2 "
    "because adding clusters gives K-means more flexibility to reduce "
    "within-cluster distances. Lower inertia always improves as K grows, "
    "so it cannot identify the best business solution by itself; "
    "silhouette scores and useful, actionable segment profiles matter too."
)

print("\nTASK 3(c): PCA")

pca = PCA(n_components=2)
pca.fit(scaled_rfm)

print("PCA explained variance ratios:", pca.explained_variance_ratio_.round(3))
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 3))

segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].copy()

segment_output.to_csv(OUTPUT / "customer_segments.csv", index=False)
print("\nSaved outputs/customer_segments.csv")
# ============================================================
# TASK 4: BUILD THE LEAKAGE-SAFE LAPSE DATASET
# ============================================================

print("\nTASK 4: BUILDING THE LAPSE DATASET")

# Only orders strictly before the cutoff may become features.
pre_cutoff = orders[
    orders["order_ts"].notna()
    & (orders["order_ts"] < CUTOFF)
].copy()

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
].copy()

# Customers must have at least one completed order before cutoff.
cohort_ids = pre_completed["customer_id"].dropna().unique()

# Label window: September 1 through September 30, 2025.
september_completed = orders[
    orders["order_ts"].notna()
    & (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < pd.Timestamp("2025-10-01"))
    & (orders["status"] == "completed")
]

september_active_ids = set(
    september_completed["customer_id"].dropna().unique()
)

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)

# Historical completed-order behaviour.
pre_completed["order_amount"] = pd.to_numeric(
    pre_completed["order_amount"], errors="coerce"
)
pre_completed["discount_pct"] = pd.to_numeric(
    pre_completed["discount_pct"], errors="coerce"
)
pre_completed["delivery_minutes"] = pd.to_numeric(
    pre_completed["delivery_minutes"], errors="coerce"
)

historical_features = pre_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", lambda x: x.sum(min_count=1)),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean"),
).reset_index()

historical_features["recency"] = (
    CUTOFF - historical_features["last_order_ts"]
).dt.days

# Cancellation rate among final-status order records before cutoff.
final_status_orders = pre_cutoff[
    pre_cutoff["status"].isin(["completed", "cancelled"])
].copy()

final_status_orders["is_cancelled"] = (
    final_status_orders["status"] == "cancelled"
).astype(int)

cancellation_features = final_status_orders.groupby(
    "customer_id"
).agg(
    final_order_count=("order_id", "nunique"),
    cancelled_order_count=("is_cancelled", "sum"),
).reset_index()

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_order_count"]
    / cancellation_features["final_order_count"].replace(0, np.nan)
)

# Favourite category based only on completed orders before cutoff.
pre_completed_with_category = pre_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
)

def most_frequent_category(series):
    values = series.dropna().astype(str)
    if values.empty:
        return "Unknown"
    return values.value_counts().index[0]

favourite_category = pre_completed_with_category.groupby(
    "customer_id"
)["category"].agg(most_frequent_category).rename(
    "favourite_category"
).reset_index()

# Join historical features and customer attributes.
model_data = cohort.merge(
    historical_features.drop(columns=["last_order_ts"]),
    on="customer_id",
    how="left",
).merge(
    cancellation_features[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left",
).merge(
    favourite_category,
    on="customer_id",
    how="left",
)

model_data["tenure_days"] = (
    CUTOFF - model_data["signup_date"]
).dt.days

model_data["favourite_category"] = (
    model_data["favourite_category"].fillna("Unknown")
)
model_data["city"] = model_data["city"].fillna("Unknown").astype(str)
model_data["gender"] = model_data["gender"].fillna("Unknown").astype(str)

# Remove any customer whose signup date is after the cutoff or missing.
model_data = model_data[
    model_data["tenure_days"].notna()
    & (model_data["tenure_days"] >= 0)
].copy()

print("Cohort size:", len(model_data))
print("Lapsed customers:", int(model_data["lapsed"].sum()))
print(
    "Lapse percentage:",
    round(model_data["lapsed"].mean() * 100, 2), "%",
)

print("\nTarget distribution:")
print(model_data["lapsed"].value_counts().sort_index())

# No segment_id, segment_name, or is_premium is included as a feature.
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age",
]
categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

feature_columns = numeric_features + categorical_features

X = model_data[feature_columns].copy()
y = model_data["lapsed"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))
print("Feature columns:", feature_columns)

# Guard against accidental future-data leakage in feature creation.
assert pre_cutoff["order_ts"].max() < CUTOFF
assert not any(
    col in feature_columns
    for col in ["segment_id", "segment_name", "is_premium"]
)

print("Leakage checks passed: features use pre-cutoff orders only.")
print("Segment labels are not used as model features.")
# ============================================================
# TASKS 5–6: PREPROCESSING, BASELINE AND MODEL TUNING
# ============================================================

print("\nTASK 5: TRAINING-ONLY PREPROCESSING")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Fit the preprocessing pipeline on training rows only.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Verify there are no missing values in the processed features.
def contains_missing(matrix):
    if hasattr(matrix, "data") and hasattr(matrix, "indices"):
        return bool(np.isnan(matrix.data).any())
    return bool(np.isnan(np.asarray(matrix)).any())

assert not contains_missing(X_train_processed)
assert not contains_missing(X_test_processed)

print("No missing values in processed training or test features.")

# Show the training-only frequency scaler mean versus full-cohort mean.
frequency_mean_train = preprocessor.named_transformers_[
    "numeric"
].named_steps["imputer"].statistics_[
    numeric_features.index("frequency")
]

# The scaler's mean is measured after median imputation.
frequency_scaler_mean = preprocessor.named_transformers_[
    "numeric"
].named_steps["scaler"].mean_[
    numeric_features.index("frequency")
]

print("Frequency mean used by scaler (training rows):",
      round(frequency_scaler_mean, 3))
print("Full-cohort raw frequency mean:",
      round(model_data["frequency"].mean(), 3))
print(
    "These means can differ because the scaler is fitted only on "
    "training rows and uses the training-only imputed feature values."
)

print("\nTASK 6(a): BASELINE")

baseline_predictions = np.zeros(len(y_test), dtype=int)

print("Baseline accuracy:",
      round(accuracy_score(y_test, baseline_predictions), 3))
print("Baseline lapsed-class precision:",
      round(precision_score(
          y_test, baseline_predictions, pos_label=1, zero_division=0
      ), 3))
print("Baseline lapsed-class recall:",
      round(recall_score(
          y_test, baseline_predictions, pos_label=1, zero_division=0
      ), 3))
print("Baseline lapsed-class F1:",
      round(f1_score(
          y_test, baseline_predictions, pos_label=1, zero_division=0
      ), 3))

print("\nTASK 6(b): LOGISTIC REGRESSION GRID SEARCH")

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

logistic_grid = GridSearchCV(
    logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
logistic_grid.fit(X_train, y_train)

best_logistic = logistic_grid.best_estimator_

print("Best logistic parameters:", logistic_grid.best_params_)
print("Best logistic CV F1:", round(logistic_grid.best_score_, 3))

print("\nTASK 6(c): RANDOM FOREST GRID SEARCH")

forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])

forest_grid = GridSearchCV(
    forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
forest_grid.fit(X_train, y_train)

best_forest = forest_grid.best_estimator_

print("Best random forest parameters:", forest_grid.best_params_)
print("Best random forest CV F1:", round(forest_grid.best_score_, 3))

print("\nTASK 6(d): UNTOUCHED TEST SET EVALUATION")

def evaluate_lapsed_model(name, model):
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("Confusion matrix (labels 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Lapsed-class precision:",
          round(precision_score(
              y_test, predictions, pos_label=1, zero_division=0
          ), 3))
    print("Lapsed-class recall:",
          round(recall_score(
              y_test, predictions, pos_label=1, zero_division=0
          ), 3))
    print("Lapsed-class F1:",
          round(f1_score(
              y_test, predictions, pos_label=1, zero_division=0
          ), 3))
    return predictions

logistic_test_predictions = evaluate_lapsed_model(
    "Tuned Logistic Regression", best_logistic
)
forest_test_predictions = evaluate_lapsed_model(
    "Tuned Random Forest", best_forest
)

# Select the model with the higher cross-validated F1.
if logistic_grid.best_score_ >= forest_grid.best_score_:
    selected_model = best_logistic
    selected_model_name = "Logistic Regression"
else:
    selected_model = best_forest
    selected_model_name = "Random Forest"

selected_test_predictions = selected_model.predict(X_test)
selected_test_f1 = f1_score(
    y_test, selected_test_predictions, pos_label=1, zero_division=0
)

print("\nSelected model:", selected_model_name)
print("Selected model test F1:", round(selected_test_f1, 3))

if selected_test_f1 < 0.55:
    print(
        "WARNING: Selected model has not met the required test F1 of 0.55. "
        "Review features and results before submission."
    )
 # ============================================================
# TASK 7: CLASS-BALANCE EXPERIMENT
# ============================================================

print("\nTASK 7: OVERSAMPLING THE TRAINING SET ONLY")

train_frame = X_train.copy()
train_frame["_target"] = y_train

majority_class = train_frame["_target"].value_counts().idxmax()
minority_class = train_frame["_target"].value_counts().idxmin()

majority_rows = train_frame[
    train_frame["_target"] == majority_class
]
minority_rows = train_frame[
    train_frame["_target"] == minority_class
]

extra_minority = minority_rows.sample(
    n=len(majority_rows) - len(minority_rows),
    replace=True,
    random_state=42,
)

balanced_train = pd.concat(
    [train_frame, extra_minority],
    ignore_index=True,
)

X_balanced = balanced_train.drop(columns="_target")
y_balanced = balanced_train["_target"].astype(int)

print("Balanced training class counts:")
print(y_balanced.value_counts().sort_index())

# Refit the best logistic regression configuration on balanced training data.
balanced_logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=logistic_grid.best_params_["model__C"],
        max_iter=1000,
    )),
])

balanced_logistic.fit(X_balanced, y_balanced)

balanced_test_predictions = balanced_logistic.predict(X_test)

print("\nOriginal logistic model — lapsed class:")
print("Precision:", round(precision_score(
    y_test, logistic_test_predictions, pos_label=1, zero_division=0
), 3))
print("Recall:", round(recall_score(
    y_test, logistic_test_predictions, pos_label=1, zero_division=0
), 3))
print("F1:", round(f1_score(
    y_test, logistic_test_predictions, pos_label=1, zero_division=0
), 3))

print("\nBalanced logistic model — same unchanged test set:")
print("Precision:", round(precision_score(
    y_test, balanced_test_predictions, pos_label=1, zero_division=0
), 3))
print("Recall:", round(recall_score(
    y_test, balanced_test_predictions, pos_label=1, zero_division=0
), 3))
print("F1:", round(f1_score(
    y_test, balanced_test_predictions, pos_label=1, zero_division=0
), 3))

print(
    "Business reading: oversampling may increase recall by identifying "
    "more customers who lapse, but it can also lower precision and cause "
    "more retention offers to go to customers who would have ordered anyway."
)

# ============================================================
# TASK 8(a): MODEL INTERPRETATION
# ============================================================

print("\nTASK 8(a): MODEL INTERPRETATION")

forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_model = best_forest.named_steps["model"]

forest_feature_names = forest_preprocessor.get_feature_names_out()
forest_importances = pd.Series(
    forest_model.feature_importances_,
    index=forest_feature_names,
).sort_values(ascending=False)

print("\nFive most important random forest features:")
print(forest_importances.head(5).round(4).to_string())

logistic_preprocessor = best_logistic.named_steps["preprocessor"]
logistic_model = best_logistic.named_steps["model"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()
logistic_coefficients = pd.Series(
    logistic_model.coef_[0],
    index=logistic_feature_names,
).sort_values(key=abs, ascending=False)

print("\nThree logistic regression coefficients with largest absolute values:")
for feature, coefficient in logistic_coefficients.head(3).items():
    direction = "increases" if coefficient > 0 else "decreases"
    print(
        f"{feature}: coefficient={coefficient:.4f}; "
        f"higher values are associated with {direction} odds of lapse, "
        "holding other model features constant."
    )

# ============================================================
# TASK 8(b): SAVE, RELOAD AND VERIFY THE SELECTED MODEL
# ============================================================

print("\nTASK 8(b): MODEL PERSISTENCE")

# Prefer a model that meets the required test F1 threshold.
forest_f1 = f1_score(
    y_test, forest_test_predictions, pos_label=1, zero_division=0
)
logistic_f1 = f1_score(
    y_test, logistic_test_predictions, pos_label=1, zero_division=0
)

if forest_f1 >= 0.55:
    selected_model = best_forest
    selected_model_name = "Random Forest"
elif logistic_f1 >= 0.55:
    selected_model = best_logistic
    selected_model_name = "Logistic Regression"
elif forest_f1 >= logistic_f1:
    selected_model = best_forest
    selected_model_name = "Random Forest"
else:
    selected_model = best_logistic
    selected_model_name = "Logistic Regression"

selected_predictions = selected_model.predict(X_test)
selected_f1 = f1_score(
    y_test, selected_predictions, pos_label=1, zero_division=0
)

model_path = OUTPUT / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)
reloaded_predictions = reloaded_model.predict(X_test)

print("Selected model:", selected_model_name)
print("Selected model test F1:", round(selected_f1, 3))
print("Saved model:", model_path)
print(
    "Reloaded predictions exactly match original:",
    np.array_equal(selected_predictions, reloaded_predictions),
)

assert np.array_equal(selected_predictions, reloaded_predictions)

# ============================================================
# TASK 8(c): PREMIUM-CUSTOMER RETENTION LIST
# ============================================================

print("\nTASK 8(c): RETENTION LIST")

# Premium is a post-hoc label from the whole-window RFM analysis.
premium_customers = set(
    rfm.loc[rfm["is_premium"], "customer_id"]
)

test_customers = model_data.iloc[
    X_test.index
] if model_data.index.equals(X_test.index) else model_data.loc[X_test.index]

# Use the original row indices from the stratified split to identify
# the corresponding test customers.
test_ids = X_test.index.map(model_data["customer_id"])
test_results = pd.DataFrame({
    "customer_id": test_ids.to_numpy(),
    "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.to_numpy(),
})

test_results = test_results[
    test_results["customer_id"].isin(premium_customers)
].sort_values("lapse_probability", ascending=False)

retention_list = test_results.head(10).copy()
retention_list["lapse_probability"] = retention_list[
    "lapse_probability"
].round(4)
retention_list["actually_lapsed"] = retention_list[
    "actually_lapsed"
].astype(int)

retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention_path = OUTPUT / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("Saved:", retention_path)
print(retention_list.to_string(index=False))
print(
    "Customers in the retention list who truly lapsed:",
    int(retention_list["actually_lapsed"].sum()),
)

print(
    "\nLIMITATION: The Premium flag is derived from whole-window RFM, "
    "including September, so this retention list is illustrative and "
    "is not a leakage-free evaluation of targeting Premium customers."
)

print("\nPart B script finished.") 
