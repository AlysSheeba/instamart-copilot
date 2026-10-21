import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "instamart.db"

tables = ["customers", "products", "orders", "tickets"]

with sqlite3.connect(DB_PATH) as conn:
    for table in tables:
        csv_path = CLEAN_DIR / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing cleaned CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"Loaded {table}: {len(df)} rows")

print(f"\nDatabase created: {DB_PATH}")
