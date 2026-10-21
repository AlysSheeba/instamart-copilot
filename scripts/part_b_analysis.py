from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01")
CUTOFF = pd.Timestamp("2025-09-01")
SEPTEMBER_END = pd.Timestamp("2025-10-01")

# ---------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------
orders = pd.read_csv(DATA / "orders.csv")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)
orders["status"] = orders["status"].astype(str).str.strip().str.lower()

for col in ["order_amount", "discount_pct", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

orders = orders.dropna(subset=["customer_id", "order_id", "order_ts"])
completed = orders[orders["status"] == "completed"].copy()

print("\nPART B: CUSTOMER SEGMENTATION AND LAPSE PREDICTION")
print("-" * 60)
print("Clean customers:", len(customers))
print("Clean orders:", len(orders))
print("Completed orders:", len(completed))

# ---------------------------------------------------------
# 2. RFM FEATURES AND K-MEANS SEGMENTATION
# ---------------------------------------------------------
# RFM uses completed orders before the 2025-10-01 snapshot.
rfm_orders = completed[completed["order_ts"] < SNAPSHOT].copy()

rfm = (
    rfm_orders.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", lambda x: x.sum(min_count=1)),
    )
    .reset_index()
)

rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
rfm = rfm.drop(columns=["last_order_ts"])
rfm = rfm.dropna(subset=["recency", "frequency"])
rfm = rfm[rfm["recency"] >= 0].copy()

# Impute missing customer monetary totals using the median
missing_monetary = rfm["monetary"].isna()
median_monetary = rfm["monetary"].median()
rfm.loc[missing_monetary, "monetary"] = median_monetary

print("Customers with imputed monetary totals:", int(missing_monetary.sum()))
print("Median monetary value used:", round(median_monetary, 2))

rfm_features = ["recency", "frequency", "monetary"]
rfm_scaler = StandardScaler()
rfm_scaled = rfm_scaler.fit_transform(rfm[rfm_features])

print("\nRFM SUMMARY")
print(rfm[rfm_features].describe().round(2))

# Compare K=2 through K=8.
k_results = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    k_results.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(rfm_scaled, labels),
    })

k_table = pd.DataFrame(k_results)
k_table.to_csv(OUTPUT / "kmeans_comparison.csv", index=False)
print("\nK-MEANS COMPARISON")
print(k_table.round(3).to_string(index=False))

# K=4 is selected for a balance of silhouette score,
# elbow behaviour, and business interpretability.
FINAL_K = 4
kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(rfm_scaled)

profile = (
    rfm.groupby("segment_id")
    .agg(
        customers=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
    )
    .reset_index()
)

premium_segment = int(
    profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]
)

# Give segment names from their observed behaviour.
def name_segment(row):
    if row["segment_id"] == premium_segment:
        return "Premium High-Value"
    if row["mean_frequency"] >= rfm["frequency"].median():
        return "Frequent Customers"
    if row["mean_recency"] >= rfm["recency"].median():
        return "At-Risk / Inactive"
    return "Developing Customers"

profile["segment_name"] = profile.apply(name_segment, axis=1)
name_lookup = dict(zip(profile["segment_id"], profile["segment_name"]))

rfm["segment_name"] = rfm["segment_id"].map(name_lookup)
rfm["is_premium"] = (rfm["segment_id"] == premium_segment).astype(int)

segments = rfm[
    [
        "customer_id", "recency", "frequency", "monetary",
        "segment_id", "segment_name", "is_premium"
    ]
].copy()
segments.to_csv(OUTPUT / "customer_segments.csv", index=False)

print("\nSEGMENT PROFILES")
print(profile.round(2).to_string(index=False))
print("\nPremium segment ID:", premium_segment)
print(
    "Inertia generally decreases as K increases because more clusters "
    "fit the data more closely. It should be considered alongside "
    "silhouette score and whether the segments are useful to the business."
)

pca = PCA(n_components=2, random_state=42)
pca_scores = pca.fit_transform(rfm_scaled)
print(
    "\nPCA explained variance ratios:",
    np.round(pca.explained_variance_ratio_, 3),
)
print(
    "PCA total variance explained:",
    round(pca.explained_variance_ratio_.sum(), 3),
)

# ---------------------------------------------------------
# 3. BUILD LAPSE LABELS AND PRE-CUTOFF FEATURES
# ---------------------------------------------------------
# Cohort: customers with at least one completed order before Sep 1.
pre_cutoff = completed[completed["order_ts"] < CUTOFF].copy()
cohort_ids = pre_cutoff["customer_id"].drop_duplicates()

# Lapsed = no completed order from Sep 1 through Sep 30.
september_completed = completed[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < SEPTEMBER_END)
]
september_active_ids = set(september_completed["customer_id"])

model_data = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()
model_data["lapsed"] = (
    ~model_data["customer_id"].isin(september_active_ids)
).astype(int)

# Historical numeric features: strictly before Sep 1.
historical_completed = pre_cutoff.copy()

historical_rfm = (
    historical_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", lambda x: x.sum(min_count=1)),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)
historical_rfm["recency"] = (
    CUTOFF - historical_rfm["last_order_ts"]
).dt.days
historical_rfm = historical_rfm.drop(columns=["last_order_ts"])

# Cancellation rate uses completed/cancelled final-status records
# strictly before the cutoff.
pre_final = orders[
    (orders["order_ts"] < CUTOFF)
    & (orders["status"].isin(["completed", "cancelled"]))
].copy()
pre_final["is_cancelled"] = (
    pre_final["status"] == "cancelled"
).astype(int)

cancel_rates = (
    pre_final.groupby("customer_id")["is_cancelled"]
    .mean()
    .rename("cancellation_rate")
    .reset_index()
)

# Favourite product category from pre-cutoff completed orders.
category_orders = historical_completed.merge(
    products[["product_id", "category"]].drop_duplicates("product_id"),
    on="product_id",
    how="left",
)
category_counts = (
    category_orders.dropna(subset=["category"])
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="order_count")
    .sort_values(
        ["customer_id", "order_count", "category"],
        ascending=[True, False, True],
    )
)
favourite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

model_data = model_data.merge(
    historical_rfm, on="customer_id", how="left"
)
model_data = model_data.merge(
    cancel_rates, on="customer_id", how="left"
)
model_data = model_data.merge(
    favourite_category, on="customer_id", how="left"
)

model_data["tenure_days"] = (
    CUTOFF - model_data["signup_date"]
).dt.days

# Do not include segment_id, segment_name, monetary from whole-window
# segments, is_premium, or September activity as model features.
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

for col in numeric_features:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

for col in categorical_features:
    model_data[col] = model_data[col].fillna("Unknown").astype(str)

# Remove customers with impossible negative tenure.
model_data = model_data[
    model_data["tenure_days"].isna()
    | (model_data["tenure_days"] >= 0)
].copy()

X = model_data[numeric_features + categorical_features].copy()
y = model_data["lapsed"].astype(int).copy()

# Confirm the features contain no post-cutoff order information.
assert "segment_id" not in X.columns
assert "segment_name" not in X.columns
assert "is_premium" not in X.columns
assert pre_cutoff["order_ts"].max() < CUTOFF

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("\nLAPSE COHORT")
print("Cohort size:", len(model_data))
print("Lapsed customers:", int(y.sum()))
print("Lapse rate (%):", round(y.mean() * 100, 2))
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Baseline: predict every customer as not lapsed.
baseline_predictions = np.zeros(len(y_test), dtype=int)
print("\nBASELINE: predict all customers as not lapsed")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
print("Lapsed-class F1:", round(f1_score(y_test, baseline_predictions), 3))

# ---------------------------------------------------------
# 4. PREPROCESSING AND TRAIN-ONLY FITTING
# ---------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
    )),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Fit preprocessing on training data only for diagnostics.
preprocessor.fit(X_train)
transformed_train = preprocessor.transform(X_train)

if hasattr(transformed_train, "toarray"):
    transformed_train_array = transformed_train.toarray()
else:
    transformed_train_array = np.asarray(transformed_train)

print(
    "\nMissing values after training-fitted preprocessing:",
    int(np.isnan(transformed_train_array).sum()),
)

numeric_scaler = (
    preprocessor.named_transformers_["numeric"]
    .named_steps["scaler"]
)
frequency_position = numeric_features.index("frequency")

print(
    "Training-fitted frequency scaler mean:",
    round(numeric_scaler.mean_[frequency_position], 3),
)
print(
    "Full cohort frequency mean:",
    round(model_data["frequency"].mean(), 3),
)
print(
    "These means can differ because the scaler is fitted on training "
    "rows only, while the cohort mean includes the held-out test rows."
)

# ---------------------------------------------------------
# 5. GRID SEARCH: LOGISTIC REGRESSION AND RANDOM FOREST
# ---------------------------------------------------------
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
logistic_grid = GridSearchCV(
    logistic_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
logistic_grid.fit(X_train, y_train)
best_logistic = logistic_grid.best_estimator_

forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )),
])
forest_grid = GridSearchCV(
    forest_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
forest_grid.fit(X_train, y_train)
best_forest = forest_grid.best_estimator_

print("\nGRID SEARCH RESULTS")
print("Logistic best parameters:", logistic_grid.best_params_)
print("Logistic best CV F1:", round(logistic_grid.best_score_, 3))
print("Random Forest best parameters:", forest_grid.best_params_)
print("Random Forest best CV F1:", round(forest_grid.best_score_, 3))

def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    print(f"\n{name}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))
    print("Accuracy:", round(accuracy_score(y_eval, predictions), 3))
    print("Precision:", round(precision_score(y_eval, predictions, zero_division=0), 3))
    print("Recall:", round(recall_score(y_eval, predictions, zero_division=0), 3))
    print("F1:", round(f1_score(y_eval, predictions, zero_division=0), 3))
    return predictions

logistic_predictions = evaluate_model(
    "LOGISTIC REGRESSION — TEST SET",
    best_logistic, X_test, y_test
)
forest_predictions = evaluate_model(
    "RANDOM FOREST — TEST SET",
    best_forest, X_test, y_test
)

# Choose the model with the stronger test F1, while reporting both.
logistic_f1 = f1_score(y_test, logistic_predictions, zero_division=0)
forest_f1 = f1_score(y_test, forest_predictions, zero_division=0)

if forest_f1 >= logistic_f1:
    selected_model = best_forest
    selected_model_name = "Random Forest"
else:
    selected_model = best_logistic
    selected_model_name = "Logistic Regression"

print("\nSELECTED MODEL:", selected_model_name)
print(
    "Selection should consider the F1 score and the business cost of "
    "missed lapsing customers versus contacting loyal customers."
)

# ---------------------------------------------------------
# 6. OVERSAMPLE LAPSED TRAINING ROWS ONLY
# ---------------------------------------------------------
# Duplicate minority-class training rows with replacement until the
# two training classes have equal counts. The test set is unchanged.
train_frame = X_train.copy()
train_frame["_target"] = y_train

class_counts = train_frame["_target"].value_counts()
majority_count = int(class_counts.max())
minority_class = int(class_counts.idxmin())
minority_rows = train_frame[
    train_frame["_target"] == minority_class
]

if len(minority_rows) > 0 and class_counts.nunique() > 1:
    extra_rows = minority_rows.sample(
        n=majority_count - len(minority_rows),
        replace=True,
        random_state=42,
    )
    balanced_train = pd.concat(
        [train_frame, extra_rows],
        ignore_index=True,
    )
else:
    balanced_train = train_frame.copy()

X_balanced = balanced_train.drop(columns="_target")
y_balanced = balanced_train["_target"].astype(int)

print("\nTRAINING COUNTS BEFORE OVERSAMPLING")
print(y_train.value_counts().sort_index().to_dict())
print("TRAINING COUNTS AFTER OVERSAMPLING")
print(y_balanced.value_counts().sort_index().to_dict())

balanced_logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=best_logistic.named_steps["model"].C,
    )),
])
balanced_logistic.fit(X_balanced, y_balanced)

print("\nOVERSAMPLED LOGISTIC REGRESSION — ORIGINAL TEST SET")
balanced_predictions = evaluate_model(
    "OVERSAMPLED LOGISTIC REGRESSION",
    balanced_logistic, X_test, y_test
)

print(
    "\nOversampling can improve recall by making the lapsed class more "
    "visible during training, but it may also increase false positives "
    "and reduce precision. All comparisons use the original unchanged test set."
)

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE AND LOGISTIC COEFFICIENTS
# ---------------------------------------------------------
# Top five Random Forest features.
forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_model = best_forest.named_steps["model"]
feature_names = forest_preprocessor.get_feature_names_out()

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": forest_model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_table.head(10).to_csv(
    OUTPUT / "random_forest_feature_importance.csv",
    index=False,
)
print("\nTOP FIVE RANDOM FOREST FEATURES")
print(importance_table.head(5).round(4).to_string(index=False))

# Show the three largest positive and negative logistic coefficients.
logistic_preprocessor = best_logistic.named_steps["preprocessor"]
logistic_model = best_logistic.named_steps["model"]
logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_table = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_model.coef_[0],
}).sort_values("coefficient")

print("\nTHREE MOST NEGATIVE LOGISTIC COEFFICIENTS")
print(coefficient_table.head(3).round(4).to_string(index=False))
print("\nTHREE MOST POSITIVE LOGISTIC COEFFICIENTS")
print(coefficient_table.tail(3).sort_values(
    "coefficient", ascending=False
).round(4).to_string(index=False))

coefficient_table.to_csv(
    OUTPUT / "logistic_coefficients.csv",
    index=False,
)

# ---------------------------------------------------------
# 8. SAVE, RELOAD, AND VERIFY THE SELECTED MODEL
# ---------------------------------------------------------
model_path = OUTPUT / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)
original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(original_predictions, reloaded_predictions)
print("\nModel saved and reloaded successfully:", model_path)
print("Reloaded model predictions match exactly.")

# ---------------------------------------------------------
# 9. ILLUSTRATIVE PREMIUM RETENTION LIST
# ---------------------------------------------------------
# Premium status comes from whole-window RFM, which includes September.
# This makes the list illustrative, not a leakage-free evaluation.
test_customer_ids = model_data.loc[X_test.index, "customer_id"].values
test_premium = model_data.loc[X_test.index, "customer_id"].map(
    segments.set_index("customer_id")["is_premium"]
).fillna(0).astype(int).values

lapse_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

retention = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.values,
    "is_premium": test_premium,
})

retention = retention[retention["is_premium"] == 1].copy()
retention = retention.sort_values(
    "lapse_probability", ascending=False
).head(10)

retention[[
    "customer_id", "lapse_probability", "actually_lapsed"
]].to_csv(OUTPUT / "retention_list.csv", index=False)

print("\nTOP PREMIUM CUSTOMERS BY PREDICTED LAPSE PROBABILITY")
print(
    retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ].round(4).to_string(index=False)
)
print(
    "Actually lapsed in this illustrative top-10 list:",
    int(retention["actually_lapsed"].sum()),
)
print(
    "LIMITATION: Premium status uses whole-window RFM, including September. "
    "The retention list is illustrative and is not a leakage-free evaluation."
)

print("\nPART B FINISHED.")
print("Check the outputs folder for generated CSV files and lapse_model.joblib.")
