import pandas as pd
from pathlib import Path

source = Path("data/raw/orders.csv")
destination = Path("data/clean/orders.csv")
destination.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

# Remove exact duplicate records only
before = len(df)
df = df.drop_duplicates().copy()
exact_duplicates_removed = before - len(df)

# Convert columns to appropriate data types
df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
df["delivery_minutes"] = pd.to_numeric(
    df["delivery_minutes"], errors="coerce"
)

# Negative delivery durations are invalid
negative_delivery = df["delivery_minutes"] < 0
negative_delivery_count = int(negative_delivery.sum())
df.loc[negative_delivery, "delivery_minutes"] = pd.NA

# Standardize text fields
for column in ["order_id", "customer_id", "product_id", "status"]:
    df[column] = df[column].astype("string").str.strip()

# Save the cleaned dataset
df.to_csv(destination, index=False)

print("Cleaned orders saved to:", destination)
print("Rows before cleaning:", before)
print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Rows after cleaning:", len(df))
print("Negative delivery times set to missing:", negative_delivery_count)
print("\nMissing values after cleaning:")
print(df.isna().sum().to_string())
print("\nOrder statuses:")
print(df["status"].value_counts(dropna=False).to_string())
