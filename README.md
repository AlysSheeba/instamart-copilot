# Instamart Customer Intelligence & Support Copilot

## Overview
This project analyzes Instamart customer, product, order, and support-ticket data using Python, data cleaning, SQLite, and SQL analysis.

## Reproduce

The project uses Python 3.11 or newer and the supplied offline `mock_llm.py`. No API key, paid service, or network access is required for model calls.

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/AlysSheeba/instamart-copilot.git
cd instamart-copilot
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate the data and prepare the database

Run the supplied data-generation script, clean the datasets, build the SQLite database, and execute the SQL analyses:

```bash
python generate_data.py
python scripts/clean_customers.py
python scripts/clean_products.py
python scripts/clean_orders.py
python scripts/clean_tickets.py
python scripts/build_database.py
python scripts/run_sql_analysis.py
```

This prepares the datasets and database and regenerates the six Part A query-result CSV files.

### 3. Run Part B: Customer analytics

```bash
python scripts/part_b_analysis.py
```

This runs customer segmentation and lapse prediction and regenerates the Part B analysis outputs.

### 4. Run Part C: Ticket triage

```bash
python scripts/part_c_triage.py
```

This runs the offline ticket-classification evaluation and regenerates the triage outputs.

### 5. Run Part D: Retrieval-augmented generation

```bash
python scripts/part_d_rag.py
```

This evaluates retrieval and answer grounding with and without chunk overlap and regenerates the RAG outputs.

### 6. Run Part D: Refund agent

The repository includes saved agent checkpoints under `state/`. To run the scenarios afresh rather than reuse completed checkpoints, remove the existing scenario checkpoints first.

```bash
python -c "from pathlib import Path; [p.unlink() for p in Path('state').glob('S*.json')]"
python scripts/part_d_agent.py
```

This runs the seven refund-agent scenarios and regenerates the agent-run and refund-ledger outputs and scenario checkpoints.

### 7. Review the results

The generated CSV and JSONL files are saved under `outputs/`. The README's Part A–D sections document the evaluation results, key metrics, and limitations.

The project uses an offline mock language model. Refund outcomes and ledger entries are simulated and do not represent real financial transactions.


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

### Policy RAG Assistant

Implemented an offline retrieval-augmented generation (RAG) evaluation pipeline using the project policy documents and `mock_llm.py`.

- Split policy documents into 40-word chunks.
- Compared retrieval with 10-word chunk overlap against retrieval without overlap.
- Retrieved the top three relevant chunks for each evaluation question.
- Evaluated retrieval using 14 questions from `data/raw/policy_eval.csv`.
- Compared source-document accuracy, expected-fact accuracy, and token usage.

### RAG Evaluation Results

| Metric | 10-word overlap | No overlap |
|---|---:|---:|
| Chunks generated | 17 | 13 |
| Evaluation questions | 14 | 14 |
| Correct source documents | 12/14 | 12/14 |
| Source accuracy | 85.71% | 85.71% |
| Expected facts matched | 14/14 | 11/14 |
| Fact accuracy | 100.00% | 78.57% |
| Total tokens | 3,096 | 3,107 |
| Mean tokens per question | 221.14 | 221.93 |

The 10-word overlap configuration achieved higher expected-fact accuracy while using slightly fewer tokens overall. Both configurations achieved the same source-document accuracy.

### Refund Agent

Implemented a tool-using refund agent with order lookup, policy search, refund calculation, refund issuance, escalation to human review, retry handling, execution limits, and JSON checkpoints. The agent records each model call, token usage, tool execution, final status, and refund outcome.

#### Seven-Scenario Evaluation

| Scenario | Status | Outcome | Refund (INR) | Model calls | Tokens |
|---|---|---|---:|---:|---:|
| S1 | COMPLETED | REFUND | 128.25 | 5 | 1,292 |
| S2 | COMPLETED | REFUND | 135.00 | 5 | 1,292 |
| S3 | COMPLETED | ESCALATE | 1,999.00 | 5 | 1,287 |
| S4 | COMPLETED | DENIED | 0.00 | 4 | 877 |
| S5 | COMPLETED | DENIED | 0.00 | 3 | 491 |
| S6 | COMPLETED | NOT_FOUND | 0.00 | 2 | 103 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK | 0.00 | 6 | 4,587 |

- Outcome matches: 7/7 (100%).
- Refund amount matches: 7/7 (100%).
- Refunds issued: 2; cases escalated to human review: 1.
- S7 repeatedly requested `search_policy` and stopped at the six-call iteration limit with status `FALLBACK_MAX_ITERATIONS`.
- The token-limit test with a 200-token budget returned `FALLBACK_TOKEN_LIMIT`.
- The zero-second timeout test returned `FALLBACK_TIMEOUT`.
- The safety tests exercised fallback statuses; they do not represent a production load or reliability test.

#### Checkpoint and Resume Test

A controlled test simulated a crash after the second tool execution. The checkpoint was saved with the tool history `get_order`, `search_policy`. Resuming continued with `calculate_refund`, `issue_refund`, without repeating the first two tool calls. The final result was `REFUND` for INR 128.25, using five model calls across the complete run. The test verified the exact four-tool sequence and refund amount.

#### ReAct and Plan-and-Execute

This implementation follows a ReAct-style loop: the model requests a tool, observes its result, and then chooses the next action. In the measured S1 run, this required five model calls and 1,292 tokens; S7 required six calls and 4,587 tokens before hitting its iteration limit. A plan-and-execute design would first create a multi-step plan and then execute its actions, potentially reducing repeated decision-making for predictable workflows, but it can also require planning overhead and may need replanning when a tool result changes the next step. No plan-and-execute baseline was run here, so its call count and token savings cannot be claimed from these results. A fair comparison would run both designs on the same scenarios and compare outcomes, model calls, tokens, and recovery behavior.

### Generated Outputs

RAG outputs:
- `outputs/rag_answers_overlap0.csv`
- `outputs/rag_answers_overlap10.csv`
- `outputs/rag_evaluation_summary.csv`
- `outputs/rag_retrieval_results.csv`

Refund-agent outputs:
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/S1.json` through `state/S7.json`

### Limitations

- RAG and refund-agent results use the offline mock model and do not establish real-world LLM performance.
- RAG accuracy was evaluated on only 14 questions; broader policy coverage is needed.
- Refund-agent evaluation used seven predefined scenarios, so it does not establish production reliability.
- Checkpoint resumption was verified with a controlled crash-after-tool test. Broader crash points, persistence failures, and duplicate side-effect prevention still need testing.
- Refund decisions, payment execution, and escalations would require appropriate validation, authorization, and monitoring in a real system.


## Limitations and Next Steps

- Verify data quality and analysis assumptions.
- Review the limitations documented for Parts A–D.
- Restore the original assignment-provided `mock_llm.py` before final submission, if required by the assignment.
- Validate the scripts and generated outputs in a clean environment.
- Consider expanding the evaluation datasets and testing error recovery, edge cases, and policy coverage.

