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

### Implementation
Implemented an offline support-ticket triage pipeline using `mock_llm.py`. The pipeline processes 164 support tickets and includes privacy redaction, input-length and empty-text guardrails, zero-shot and few-shot prompts, retry handling, output validation, token-budget enforcement, and ticket routing.

### Privacy and Guardrails
- Read ticket data while preserving empty text values.
- Redacted email addresses, phone numbers, and order IDs before sending ticket text to the model.
- Applied 278 substitutions: 55 emails, 91 phone numbers, and 132 order IDs. No matching sensitive-data patterns remained after redaction.
- Skipped 4 empty tickets and flagged 4 long tickets without model classification.
- Retried transient errors with exponential backoff and jitter. All 19 transient errors in each full run recovered successfully.
- Validated JSON responses and accepted a single unambiguous category keyword when the response was not valid JSON. Invalid or unsupported categories were flagged for human review.

### Classification Results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy among classified tickets | 85.7% | 100.0% |
| Total model tokens | 12,208 | 20,476 |
| Tokens per classified ticket | 83.05 | 139.29 |

The few-shot prompt improved accuracy by 14.3 percentage points, but used 8,268 additional tokens. In the zero-shot confusion table, 11 fraud tickets and 10 technical tickets were incorrectly classified as billing. The few-shot examples improved category discrimination in this mock evaluation. These results reflect the deterministic mock classifier and should not be interpreted as evidence of real-world LLM performance.

Each full run produced 135 `classified_json`, 12 `classified_keyword`, and 9 `flagged_invalid_output` results among the 156 tickets eligible for model classification. The remaining 8 tickets were handled by the empty-text and long-input guardrails.

### Token Budget and Routing
A zero-shot budget demonstration used a nominal 6,000-token budget. The run consumed 6,040 tokens before stopping further calls and marked 79 tickets as `budget_exhausted`. This small overshoot occurs because the implementation checks the accumulated token count before each call; an individual call can take the total over budget.

The final routing output uses the few-shot run with a 100,000-token budget:
- Tickets requiring review—including empty, long, invalid-output, and budget-exhausted tickets—are routed to `human_review` with `NORMAL` priority.
- Fraud tickets are routed to `security_team` with `HIGH` priority.
- Other tickets belonging to premium customers are routed to `premium_desk` with `HIGH` priority.
- Remaining valid classifications are routed to `support_queue` with `NORMAL` priority.

### Output Files
- `outputs/triage_results.csv` — final few-shot classification and routing results for all 164 tickets.
- `outputs/triage_budget_demo.csv` — zero-shot token-budget demonstration for all 164 tickets.

The output files contain the required routing fields: `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.

### Limitations
- The project uses an offline mock classifier rather than a live language model.
- Token counts and classification performance are specific to the mock implementation.
- Keyword-based classification and fallback parsing may fail on ambiguous or complex customer messages.
- Tickets requiring human review should be assessed by a support agent before a final decision is made.

## Part D: Policy RAG Assistant and Refund Agent
Implementation and evaluation pending.

## Limitations and Next Steps
- Verify data quality and analysis assumptions.
- Complete and evaluate Parts B, C, and D.
- Restore the original assignment-provided `mock_llm.py` before final submission.
- Document the methods and evaluation results for each completed part.
