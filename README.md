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

Implemented customer segmentation and lapse prediction using Python and scikit-learn.

### Customer Segmentation
- Calculated customer Recency, Frequency, and Monetary (RFM) metrics.
- Handled two missing customer monetary totals using median imputation.
- Compared K-means clustering solutions for K = 2 to 8 using inertia and silhouette scores.
- Selected K = 4, with a silhouette score of approximately 0.526.
- Applied PCA with two components, explaining approximately 85.2% of the variance.

### Lapse Prediction
- Built a customer lapse prediction cohort using historical features and a September lapse target.
- Compared Logistic Regression and Random Forest using a held-out test set.
- Selected Logistic Regression based on its test-set F1 score of 0.640.
- Evaluated training-only random oversampling; the oversampled Logistic Regression achieved an F1 score of 0.697 and recall of 0.854 on the original test set.
- Saved the trained model to `outputs/lapse_model.joblib` and verified that reloaded predictions matched the original predictions.
- Generated feature importance, model coefficient, cluster comparison, customer segment, and retention-list outputs.

### Generated Outputs
- `outputs/customer_segments.csv`
- `outputs/kmeans_comparison.csv`
- `outputs/lapse_model.joblib`
- `outputs/logistic_coefficients.csv`
- `outputs/random_forest_feature_importance.csv`
- `outputs/retention_list.csv`

### Limitation
The illustrative retention list uses premium-customer status calculated from the full observation window, including September. It should not be interpreted as a leakage-free retention evaluation.

## Part C: Support-Ticket Triage and Privacy
Implementation and evaluation pending.

## Part D: Policy RAG Assistant and Refund Agent
Implementation and evaluation pending.

## Limitations and Next Steps
- Verify data quality and analysis assumptions.
- Complete and evaluate Parts B, C, and D.
- Restore the original assignment-provided `mock_llm.py` before final submission.
- Document the methods and evaluation results for each completed part.
