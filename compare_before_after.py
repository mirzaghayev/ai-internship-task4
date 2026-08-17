from run_eval import run_eval
from rag_utils import SYSTEM_PROMPT
from few_shot_fix import SYSTEM_PROMPT_V2

print("Running BEFORE (original SYSTEM_PROMPT) on the HOLDOUT split...\n")
rows_before, summary_before = run_eval(
    split="holdout", system_prompt=SYSTEM_PROMPT, out_prefix="holdout_before"
)

print("\nRunning AFTER (few-shot fixed SYSTEM_PROMPT_V2) on the HOLDOUT split...\n")
rows_after, summary_after = run_eval(
    split="holdout", system_prompt=SYSTEM_PROMPT_V2, out_prefix="holdout_after"
)

print("\n=== BEFORE vs AFTER (held-out set, unseen during fix design) ===")
print(f"{'metric':<28}{'before':<15}{'after':<15}")
for key in summary_before:
    print(f"{key:<28}{str(summary_before[key]):<15}{str(summary_after.get(key, '-')):<15}")

# Specifically call out the refusal-format questions, since that's what the fix targets
print("\n--- out_of_scope items only (what the fix specifically targets) ---")
before_refusal = [r for r in rows_before if r["category"] == "out_of_scope"]
after_refusal = [r for r in rows_after if r["category"] == "out_of_scope"]
for b, a in zip(before_refusal, after_refusal):
    print(f"[{b['id']}] before={b['verdict']:<5} after={a['verdict']:<5}")
    print(f"    before_answer: {b['actual_answer'][:100]!r}")
    print(f"    after_answer:  {a['actual_answer'][:100]!r}")