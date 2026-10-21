import pandas as pd
from pathlib import Path

source = Path("data/raw/customers.csv")
destination = Path("data/clean/customers.csv")
destination.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

# Standardize city names
df["city"] = df["city"].astype("string").str.strip().str.title()

# Standardize gender values
gender_map = {
    "M": "Male", "MALE": "Male",
    "F": "Female", "FEMALE": "Female"
}
df["gender"] = (
    df["gender"].astype("string").str.strip().str.upper()
    .map(gender_map).fillna("Unknown")
)

# Convert invalid ages to missing, then impute with valid-age median
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[~df["age"].between(18, 100), "age"] = pd.NA
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age).round().astype("int64")

# Save cleaned data
df.to_csv(destination, index=False)

print("Cleaned customer data saved to:", destination)
print("Rows:", len(df))
print("\nMissing values after cleaning:")
print(df.isna().sum().to_string())
print("\nCity categories:")
print(sorted(df["city"].unique()))
print("\nGender categories:")
print(df["gender"].value_counts().to_string())
print("\nAge range:", df["age"].min(), "to", df["age"].max())
