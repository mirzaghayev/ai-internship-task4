import argparse
import csv
import json
from pathlib import Path

from eval_pipeline import run_pipeline_instrumented
from judge import judge_open_ended, judge_refusal

FILE_PATH = "test_file.pdf"
TEST_SET_PATH = "test_set.json"

# --- Cost estimation --------------------------------------------------------
# Groq's per-token pricing changes and varies by model -- these are
# PLACEHOLDER rates. Check https://groq.com/pricing (or your Groq console)
# for llama-3.1-8b-instant's current $/1M input and output tokens and
# update the two constants below before trusting the cost numbers.
PRICE_PER_M_INPUT_TOKENS = 0.05
PRICE_PER_M_OUTPUT_TOKENS = 0.08


def estimate_cost(prompt_tokens, completion_tokens):
    return (
        (prompt_tokens / 1_000_000) * PRICE_PER_M_INPUT_TOKENS
        + (completion_tokens / 1_000_000) * PRICE_PER_M_OUTPUT_TOKENS
    )


def run_eval(split=None, system_prompt=None, out_prefix="results", verbose=True):
    """split: None (all items), 'dev', or 'holdout'."""
    test_set = json.loads(Path(TEST_SET_PATH).read_text())
    if split:
        test_set = [t for t in test_set if t["split"] == split]

    rows = []
    for item in test_set:
        result = run_pipeline_instrumented(
            FILE_PATH, item["question"], system_prompt=system_prompt
        )

        if item["category"] == "out_of_scope":
            verdict, reasoning = judge_refusal(result["answer"])
        else:
            verdict, reasoning = judge_open_ended(
                item["question"], item["expected_answer"], result["answer"]
            )

        cost = estimate_cost(result["prompt_tokens"], result["completion_tokens"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "split": item["split"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "actual_answer": result["answer"],
            "verdict": verdict,
            "judge_reasoning": reasoning,
            "retrieval_latency_s": round(result["retrieval_latency_s"], 3),
            "generation_latency_s": round(result["generation_latency_s"], 3),
            "total_latency_s": round(result["total_latency_s"], 3),
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "estimated_cost_usd": round(cost, 6),
        }
        rows.append(row)
        if verbose:
            print(f"[{item['id']:<12}] {item['category']:<20} verdict={verdict}  ({reasoning})")

    if not rows:
        print("No test items matched the given split.")
        return [], {}

    out_csv = f"{out_prefix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_partial = sum(1 for r in rows if r["verdict"] == "PARTIAL")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    accuracy = n_pass / n

    summary = {
        "n_questions": n,
        "pass": n_pass,
        "partial": n_partial,
        "fail": n_fail,
        "accuracy_pass_rate": round(accuracy, 3),
        "avg_latency_s": round(sum(r["total_latency_s"] for r in rows) / n, 3),
        "avg_retrieval_latency_s": round(sum(r["retrieval_latency_s"] for r in rows) / n, 3),
        "avg_generation_latency_s": round(sum(r["generation_latency_s"] for r in rows) / n, 3),
        "avg_total_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "avg_estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in rows) / n, 6),
    }

    if verbose:
        print(f"\n=== SUMMARY ({out_prefix}) ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"\nDetailed rows written to {out_csv}")

    Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout"], default=None)
    parser.add_argument("--out-prefix", default="results")
    args = parser.parse_args()
    run_eval(split=args.split, out_prefix=args.out_prefix)