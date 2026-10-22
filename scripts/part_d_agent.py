import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

DB_PATH = ROOT / "instamart.db"
SCENARIO_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
POLICY_DIR = ROOT / "data" / "policies"
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "outputs"

MAX_STEPS = 6
MAX_TOKENS = 6000
MAX_SECONDS = 10
MAX_RETRIES = 3


def read_scenarios():
    import csv
    with SCENARIO_PATH.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT order_id, customer_id, status, order_amount, "
            "delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    return dict(row) if row else {"error": "order_not_found"}


def search_policy(query):
    matches = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if "refund" in query.lower() and "refund" in text.lower():
            matches.append({"source": path.name, "text": text})
    return {"policies": matches}


def calculate_refund(order_amount, delivery_minutes):
    amount = round(float(order_amount), 2)
    minutes = float(delivery_minutes)
    refund = amount if minutes > 45 else 0.0
    return {"refund_inr": round(refund, 2)}


def escalate_to_human(order_id, refund_inr):
    return {
        "status": "ESCALATED",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "message": "Human support lead approval required.",
    }


def issue_refund(order_id, refund_inr):
    return {
        "status": "ISSUED",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }


def execute_tool(name, args):
    if name == "get_order":
        return get_order(args["order_id"])
    if name == "search_policy":
        return search_policy(args.get("query", "refund policy"))
    if name == "calculate_refund":
        return calculate_refund(
            args["order_amount"], args["delivery_minutes"]
        )
    if name == "escalate_to_human":
        return escalate_to_human(args["order_id"], args["refund_inr"])
    if name == "issue_refund":
        return issue_refund(args["order_id"], args["refund_inr"])
    return {"error": "unknown_tool", "tool": name}


def parse_tool_arguments(raw):
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def run_scenario(scenario, ledger, token_limit=MAX_TOKENS, time_limit_seconds=MAX_SECONDS, resume=False, crash_after_tool=None):
    scenario_id = scenario["scenario_id"]
    checkpoint = STATE_DIR / f"{scenario_id}.json"

    saved = None
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("complete"):
            return saved["result"]

    start = time.monotonic()
    result = "FALLBACK"
    status = "COMPLETED"

    if resume and saved:
        messages = saved["messages"]
        token_total = saved.get("tokens_used", saved.get("tokens", 0))
        steps = saved.get("steps", 0)
        trace = saved.get("trace", [])
        order_id = saved.get("order_id")
        refund_amount = saved.get("refund_amount", 0.0)
        tools_executed = saved.get("tools_executed", [])
    else:
        user_message = scenario["message"]
        messages = [
            {
                "role": "system",
                "content": (
                    "AGENT. Follow refund policy. Use tools to verify orders, "
                    "check eligibility, calculate refunds, and escalate refunds "
                    "above INR 1000. Never invent order data."
                ),
            },
            {"role": "user", "content": user_message},
        ]
        token_total = 0
        steps = 0
        trace = []
        order_id = None
        refund_amount = 0.0
        tools_executed = []

    while steps < MAX_STEPS:
        if time.monotonic() - start >= time_limit_seconds:
            result = "FALLBACK"
            status = "FALLBACK_TIMEOUT"
            break
        if token_total >= token_limit:
            result = "FALLBACK"
            status = "FALLBACK_TOKEN_LIMIT"
            break

        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = mock_llm.chat(messages, tools=[])
                break
            except mock_llm.TransientLLMError:
                if attempt == MAX_RETRIES - 1:
                    result = "FALLBACK"
                else:
                    time.sleep(0.05 * (attempt + 1))

        if response is None:
            break

        steps += 1
        token_total += response.usage.total_tokens
        message = response.choices[0].message
        trace.append({
            "step": steps,
            "finish_reason": response.choices[0].finish_reason,
            "tokens": response.usage.total_tokens,
        })

        # Stop before executing a tool if this response exceeds the limit.
        if token_total > token_limit:
            result = "FALLBACK"
            status = "FALLBACK_TOKEN_LIMIT"
            break

        if not message.tool_calls:
            text = message.content or ""
            if text.startswith("REFUND:"):
                result = "REFUND"
                try:
                    refund_amount = float(text.split("INR", 1)[1].split()[0])
                except (IndexError, ValueError):
                    refund_amount = 0.0
            elif text.startswith("ESCALATE:"):
                result = "ESCALATE"
                refund_amount = float(scenario["expected_refund_inr"])
            elif text.startswith("NOT_FOUND:"):
                result = "NOT_FOUND"
            elif text.startswith("DENIED:"):
                result = "DENIED"
            else:
                result = "FALLBACK"
            break

        tool_call = message.tool_calls[0]
        tool_name = tool_call.name
        args = parse_tool_arguments(tool_call.arguments)

        if tool_name == "get_order":
            order_id = args.get("order_id")
        tool_result = execute_tool(tool_name, args)
        tools_executed.append(tool_name)
        if tool_name == "issue_refund":
            refund_amount = float(args["refund_inr"])
        elif tool_name == "escalate_to_human":
            refund_amount = float(args["refund_inr"])

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "name": tool_name,
                "arguments": json.dumps(args),
            }],
        })
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(tool_result),
        })

        checkpoint_data = {
            "scenario_id": scenario_id,
            "source_scenario_id": scenario.get(
                "source_scenario_id", scenario_id
            ),
            "complete": False,
            "messages": messages,
            "steps": steps,
            "tokens": token_total,
            "tokens_used": token_total,
            "trace": trace,
            "order_id": order_id,
            "refund_amount": refund_amount,
            "tools_executed": tools_executed,
        }
        save_json(checkpoint, checkpoint_data)

        # Simulate a crash only after the checkpoint is safely written.
        if (
            crash_after_tool is not None
            and len(tools_executed) == crash_after_tool
        ):
            raise RuntimeError(
                f"Simulated crash after tool {crash_after_tool}: {tool_name}"
            )

    elapsed = round(time.monotonic() - start, 4)
    expected = scenario["expected_outcome"]
    expected_amount = round(float(scenario["expected_refund_inr"]), 2)

    # The mock LLM may return a tool result that ends the workflow.
    if steps >= MAX_STEPS and status == "COMPLETED":
        status = "FALLBACK_MAX_ITERATIONS"
        result = "FALLBACK"

    record = {
        "scenario_id": scenario_id,
        "customer_id": scenario["customer_id"],
        "order_id": order_id,
        "actual_outcome": result,
        "status": status,
        "final_answer": (
            f"Human handoff required: {status}."
            if status.startswith("FALLBACK") else result
        ),
        "llm_calls": steps,
        "tools_executed": tools_executed,
        "expected_outcome": expected,
        "refund_inr": round(refund_amount, 2),
        "expected_refund_inr": expected_amount,
        "outcome_match": result == expected,
        "refund_match": round(refund_amount, 2) == expected_amount,
        "steps": steps,
        "tokens_used": token_total,
        "elapsed_seconds": elapsed,
        "trace": trace,
    }

    ledger.append({
        "scenario_id": scenario_id,
        "order_id": order_id,
        "outcome": result,
        "refund_inr": round(refund_amount, 2),
        "ledger_status": (
            "ISSUED" if result == "REFUND"
            else "PENDING_HUMAN" if result == "ESCALATE"
            else "NO_PAYMENT"
        ),
    })

    save_json(checkpoint, {
        "scenario_id": scenario_id,
        "source_scenario_id": scenario.get(
            "source_scenario_id", scenario_id
        ),
        "complete": True,
        "result": record,
        "messages": messages,
        "steps": steps,
        "tokens": token_total,
        "tokens_used": token_total,
        "trace": trace,
        "order_id": order_id,
        "refund_amount": refund_amount,
        "tools_executed": tools_executed,
    })
    return record


def resume_agent(run_id, token_limit=MAX_TOKENS,
                 time_limit_seconds=MAX_SECONDS):
    """Resume an incomplete run from its most recent saved checkpoint."""
    checkpoint = STATE_DIR / f"{run_id}.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for run: {run_id}")

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    if saved.get("complete"):
        return saved["result"]

    source_id = saved.get("source_scenario_id", saved["scenario_id"])
    source = next(
        (item for item in read_scenarios()
         if item["scenario_id"] == source_id),
        None,
    )
    if source is None:
        raise ValueError(f"Scenario not found: {source_id}")

    scenario = dict(source)
    scenario["scenario_id"] = run_id
    scenario["source_scenario_id"] = source_id
    return run_scenario(
        scenario,
        [],
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        resume=True,
    )


def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = read_scenarios()
    results = []
    ledger = []

    mock_llm.reset()
    for scenario in scenarios:
        result = run_scenario(scenario, ledger)
        results.append(result)
        print(
            f"{result['scenario_id']}: "
            f"{result['actual_outcome']} "
            f"(expected {result['expected_outcome']}) "
            f"refund=INR {result['refund_inr']:.2f} "
            f"tokens={result['tokens_used']}"
        )

    # Rebuild the ledger from all results, including recovered checkpoints.
    ledger = []
    for result in results:
        outcome = result["actual_outcome"]
        ledger.append({
            "scenario_id": result["scenario_id"],
            "order_id": result.get("order_id"),
            "outcome": outcome,
            "refund_inr": round(float(result["refund_inr"]), 2),
            "ledger_status": (
                "ISSUED" if outcome == "REFUND"
                else "PENDING_HUMAN" if outcome == "ESCALATE"
                else "NO_PAYMENT"
            ),
        })

    with (OUTPUT_DIR / "agent_runs.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "refund_ledger.jsonl").open("w", encoding="utf-8") as f:
        for item in ledger:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nAgent evaluation")
    print(f"Scenarios: {len(results)}")
    print(f"Outcome matches: {sum(r['outcome_match'] for r in results)}/{len(results)}")
    print(f"Refund matches: {sum(r['refund_match'] for r in results)}/{len(results)}")
    print(f"Agent runs: {OUTPUT_DIR / 'agent_runs.jsonl'}")
    print(f"Refund ledger: {OUTPUT_DIR / 'refund_ledger.jsonl'}")
    print(f"Checkpoints: {STATE_DIR}")


if __name__ == "__main__":
    main()
