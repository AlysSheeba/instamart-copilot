import pandas as pd
from pathlib import Path

input_file = Path("data/raw/tickets.csv")
output_file = Path("data/clean/tickets.csv")

df = pd.read_csv(input_file)

print("Rows before cleaning:", len(df))

# Standardize text fields without changing ticket meaning
for col in ["ticket_id", "customer_id", "order_id", "true_category"]:
    df[col] = df[col].astype("string").str.strip()

df["text"] = df["text"].fillna("").astype("string").str.strip()

# Convert blank order IDs to missing values
df["order_id"] = df["order_id"].replace("", pd.NA)

# Remove exact duplicate rows only
before = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", before - len(df))

# Remove rows without essential identifiers or category
before = len(df)
df = df.dropna(subset=["ticket_id", "customer_id", "true_category"])
print("Rows removed for missing essential fields:", before - len(df))

output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("\nCleaned tickets saved to:", output_file)
print("Rows after cleaning:", len(df))
print("\nMissing values after cleaning:")
print(df.isna().sum().to_string())
print("\nEmpty ticket texts:", int(df["text"].eq("").sum()))
print("\nTicket categories:")
print(df["true_category"].value_counts().to_string())
