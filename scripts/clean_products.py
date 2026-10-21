import pandas as pd
from pathlib import Path

input_file = Path("data/raw/products.csv")
output_file = Path("data/clean/products.csv")

df = pd.read_csv(input_file)

print("Rows before cleaning:", len(df))

# Standardize text fields
for col in ["product_id", "product_name", "category"]:
    df[col] = df[col].astype("string").str.strip()

# Convert prices to numeric values
df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

# Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", before - len(df))

# Keep only records with valid IDs, names, categories, and positive prices
before = len(df)
df = df.dropna(subset=["product_id", "product_name", "category", "price_inr"])
df = df[
    (df["product_id"] != "") &
    (df["product_name"] != "") &
    (df["category"] != "") &
    (df["price_inr"] > 0)
]
print("Invalid rows removed:", before - len(df))

df["price_inr"] = df["price_inr"].astype(int)

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("\nCleaned products saved to:", output_file)
print("Rows after cleaning:", len(df))
print("Missing values after cleaning:")
print(df.isna().sum().to_string())
print("\nCategories:")
print(sorted(df["category"].unique().tolist()))
print("Price range: ₹", df["price_inr"].min(), "- ₹", df["price_inr"].max())
