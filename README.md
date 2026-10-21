# Instamart Customer Intelligence & Support Copilot

## Overview
This project analyzes Instamart customer, product, order, and support-ticket data using Python, data cleaning, SQLite, and SQL analysis.

## Reproduce
1. Install Python 3.
2. Install dependencies using: `pip install -r requirements.txt`

## Part A: Data Cleaning and SQL Analysis
- Generated raw datasets for customers, products, orders, and tickets.
- Cleaned the datasets using Python scripts.
- Built the SQLite database: `instamart.db`.
- Executed six SQL analysis queries.
- Saved results in the `outputs/` folder.

### Main Findings
- Completed order records: 11,110
- Cancelled order records: 1,199
- Pending order records: 656
- Total sales from completed order records: INR 2,968,622.15
- Additional analyses cover top-selling products, city-wise sales, average delivery times, and customers with the most completed order records.

## Part B: Customer Segmentation and Lapse Prediction
Implementation and evaluation pending.

## Part C: Support-Ticket Triage and Privacy
Implementation and evaluation pending.

## Part D: Policy RAG Assistant and Refund Agent
Implementation and evaluation pending.

## Limitations and Next Steps
- Verify data quality and analysis assumptions.
- Complete and evaluate Parts B, C, and D.
- Restore the original assignment-provided `mock_llm.py` before final submission.
- Document the methods and evaluation results for each completed part.
