"""Part C: Support Ticket Triage with Guardrails.

Uses the supplied offline mock_llm. No network or API key is required.
Run from the repository root: python scripts/part_c_triage.py
"""

import json
import random
import re
import time
from collections import Counter

import pandas as pd
import mock_llm


TICKETS_FILE = "data/raw/tickets.csv"
SEGMENTS_FILE = "outputs/customer_segments.csv"
OUTPUT_DIR = "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Apply these patterns in the required order.
REDACTION_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """CLASSIFY each support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in this exact form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or extra text."""

FEW_SHOT_PROMPT = """CLASSIFY each support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return JSON only in this exact form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include markdown or extra text.
EXAMPLES:
Ticket: My card was charged without permission.
Reply: {"category": "fraud", "reason": "unauthorized transaction"}
Ticket: My invoice shows the wrong amount.
Reply: {"category": "billing", "reason": "incorrect invoice amount"}
Ticket: My parcel has not arrived.
Reply: {"category": "delivery", "reason": "parcel not received"}"""


def est_tokens(text):
    """Deterministic rough estimate used only for the input-length guardrail."""
    return len(re.findall(r"\S+", str(text))) + len(str(text)) // 4


def redact_text(text):
    """Return redacted text and substitution counts."""
    redacted = text
    counts = {}
    for name, pattern, placeholder in REDACTION_PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        counts[name] = count
    return redacted, counts


def assert_redacted(text):
    """Raise an assertion error if any required pattern remains."""
    for name, pattern, _placeholder in REDACTION_PATTERNS:
        assert not pattern.search(text), f"Unredacted {name} found in prompt"


def call_with_retries(ticket_id, messages):
    """Retry transient failures up to three times, logging each retry."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} wait={wait:.5f}s"
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable retry state")


def classify_reply(reply_text):
    """Validate JSON first, then accept an unambiguous category word."""
    try:
        parsed = json.loads(reply_text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(reply_text).lower())
        found = set(words) & CATEGORIES
        if len(found) == 1:
            return next(iter(found)), "classified_keyword"
        return "", "flagged_invalid_output"

    if isinstance(parsed, dict):
        category = parsed.get("category")
        if isinstance(category, str) and category in CATEGORIES:
            return category, "classified_json"

    return "", "flagged_invalid_output"


def load_inputs():
    # keep_default_na=False preserves empty ticket text as an empty string.
    tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)

    required = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(f"Ticket file is missing columns: {sorted(missing)}")

    if "customer_id" not in segments.columns:
        raise ValueError("Customer segments file has no customer_id column")

    return tickets, segments


def premium_lookup(segments):
    """Return customer -> (premium flag, segment name)."""
    lookup = {}

    for _, row in segments.iterrows():
        customer_id = str(row["customer_id"])
        segment_name = str(row.get("segment_name", ""))

        raw_premium = row.get("is_premium", False)
        premium = str(raw_premium).strip().lower() in {
            "true", "1", "yes", "y"
        }

        # If the file lacks is_premium, use the Part B premium segment label.
        if "is_premium" not in segments.columns:
            premium = "premium" in segment_name.lower()

        lookup[customer_id] = (premium, segment_name)

    return lookup


def route_ticket(status, category, premium):
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }:
        return "NORMAL", "human_review"

    if category == "fraud":
        return "HIGH", "security_team"

    if premium:
        return "HIGH", "premium_desk"

    return "NORMAL", "support_queue"


def run_pipeline(tickets, prompt, token_budget=None, run_name="run"):
    """Process tickets in file order and optionally enforce a token budget."""
    mock_llm.reset()

    rows = []
    total_tokens = 0
    retries = 0
    status_counts = Counter()

    for _, ticket in tickets.iterrows():
        ticket_id = str(ticket["ticket_id"])
        customer_id = str(ticket["customer_id"])
        original_text = str(ticket["text"])

        category = ""
        status = ""
        ticket_tokens = 0

        if not original_text.strip():
            status = "skipped_empty"

        elif est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            redacted_text, _ = redact_text(original_text)
            assert_redacted(redacted_text)

            if token_budget is not None and total_tokens >= token_budget:
                status = "budget_exhausted"

            else:
                # The user message content must be exactly the redacted text.
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": redacted_text},
                ]

                # Assert privacy against the exact content about to be sent.
                assert messages[-1]["content"] == redacted_text
                assert_redacted(messages[-1]["content"])

                # Each successful response contributes its actual usage.
                # The mock's transient failures do not return usage.
                response = call_with_retries(ticket_id, messages)
                usage = getattr(response, "usage", None)
                ticket_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                total_tokens += ticket_tokens

                reply = response.choices[0].message.content
                category, status = classify_reply(reply)

        status_counts[status] += 1
        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "true_category": str(ticket["true_category"]),
            "tokens_used": ticket_tokens,
        })

    print(f"\n--- {run_name} ---")
    print("Status counts:", dict(status_counts))
    print("Total model tokens:", total_tokens)
    print("Recovered retries:", "see RETRY log lines above")

    return rows, total_tokens


def classification_metrics(rows, total_tokens, run_name):
    classified = [
        row for row in rows
        if row["status"] in {"classified_json", "classified_keyword"}
    ]
    correct = sum(
        row["category"] == row["true_category"] for row in classified
    )
    count = len(classified)
    accuracy = correct / count if count else 0.0

    print(f"\n{run_name} metrics")
    print(f"Classified tickets: {count}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per classified ticket: {total_tokens / count:.2f}" if count else
          "Tokens per classified ticket: N/A")

    return classified, accuracy, count


def print_confusion_table(classified):
    if not classified:
        print("No classified tickets for confusion table.")
        return

    actual = [c for c in CATEGORIES]
    table = pd.crosstab(
        pd.Categorical(
            [row["true_category"] for row in classified],
            categories=actual,
        ),
        pd.Categorical(
            [row["category"] for row in classified],
            categories=actual,
        ),
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    print("\nZero-shot confusion table:")
    print(table)


def save_routing_output(rows, segments, filename):
    premium = premium_lookup(segments)
    output = []

    for row in rows:
        is_premium, segment_name = premium.get(
            row["customer_id"], (False, "")
        )

        priority, route = route_ticket(
            row["status"], row["category"], is_premium
        )

        output.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": row["status"],
            "category": row["category"],
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"],
        })

    columns = [
        "ticket_id", "customer_id", "status", "category",
        "segment_name", "priority", "route", "tokens_used",
    ]
    pd.DataFrame(output, columns=columns).to_csv(filename, index=False)
    print(f"Saved {len(output)} rows to {filename}")


def report_redactions(tickets):
    totals = Counter()
    redacted_texts = []

    for text in tickets["text"].astype(str):
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)
        totals.update(counts)

    for name, pattern, _placeholder in REDACTION_PATTERNS:
        remaining = sum(bool(pattern.search(text)) for text in redacted_texts)
        assert remaining == 0, f"{remaining} texts still contain {name}"

    print("\nRedaction substitutions across all tickets:")
    for name in ("email", "phone", "order_id"):
        print(f"{name}: {totals[name]}")
    print(f"Total substitutions: {sum(totals.values())}")
    print("Required patterns remaining after redaction: 0")


def main():
    tickets, segments = load_inputs()
    report_redactions(tickets)

    # Compare both prompts without a token budget.
    zero_rows, zero_tokens = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, None, "zero-shot"
    )
    zero_classified, zero_accuracy, zero_count = classification_metrics(
        zero_rows, zero_tokens, "Zero-shot"
    )
    print_confusion_table(zero_classified)

    few_rows, few_tokens = run_pipeline(
        tickets, FEW_SHOT_PROMPT, None, "few-shot"
    )
    few_classified, few_accuracy, few_count = classification_metrics(
        few_rows, few_tokens, "Few-shot"
    )

    # Budget demonstration uses zero-shot, as required.
    budget_rows, budget_tokens = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, TOKEN_BUDGET_DEMO, "budget-demo"
    )
    budget_path = f"{OUTPUT_DIR}/triage_budget_demo.csv"
    save_routing_output(budget_rows, segments, budget_path)
    exhausted = sum(
        row["status"] == "budget_exhausted" for row in budget_rows
    )
    print(f"Budget demo limit: {TOKEN_BUDGET_DEMO}")
    print(f"Budget-exhausted tickets: {exhausted}")
    print(f"Tokens consumed before budget stops further calls: {budget_tokens}")

    # Final few-shot run uses the large budget and creates the required output.
    final_rows, final_tokens = run_pipeline(
        tickets, FEW_SHOT_PROMPT, TOKEN_BUDGET_FINAL, "final-few-shot"
    )
    final_path = f"{OUTPUT_DIR}/triage_results.csv"
    save_routing_output(final_rows, segments, final_path)

    # Compare the no-budget runs using measured results, not hardcoded values.
    print("\n--- Prompt cost comparison ---")
    print(f"Zero-shot: {zero_count} classified, accuracy {zero_accuracy:.1%}, "
          f"{zero_tokens} tokens")
    print(f"Few-shot: {few_count} classified, accuracy {few_accuracy:.1%}, "
          f"{few_tokens} tokens")
    print("Few-shot accuracy improvement (percentage points): "
          f"{(few_accuracy - zero_accuracy) * 100:.2f}")
    print("Additional tokens per classified ticket: "
          f"{(few_tokens / few_count - zero_tokens / zero_count):.2f}"
          if zero_count and few_count else "N/A")
    print("\nReview the printed metrics before documenting the README results.")


if __name__ == "__main__":
    main()
