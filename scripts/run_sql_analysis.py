import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "instamart.db"
SQL_PATH = ROOT / "sql" / "analysis_queries.sql"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

queries = [
    q.strip()
    for q in SQL_PATH.read_text(encoding="utf-8-sig").split(";")
    if q.strip()
]

with sqlite3.connect(DB_PATH) as conn:
    for number, query in enumerate(queries, start=1):
        cursor = conn.execute(query)
        if cursor.description is None:
            continue

        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        output_path = OUTPUT_DIR / f"part_a_query_{number}.csv"

        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)

        print(f"Saved {output_path.relative_to(ROOT)} ({len(rows)} rows)")

print("\nPart A SQL results exported successfully.")
