import json
from rag_utils import client

JUDGE_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile deprecated 08/16/26; official replacement
REFUSAL_STRING = "i don't know based on the provided documents"

JUDGE_SYSTEM_PROMPT = """You are a strict grading assistant for a RAG system's answers.
You will be given a QUESTION, an EXPECTED ANSWER (or expected criteria), and an
ACTUAL ANSWER produced by the system under test.

Grading rules:
- Judge factual correctness against the EXPECTED ANSWER / criteria only.
- Do NOT reward longer, more detailed, or more confident-sounding answers.
  A short correct answer and a long correct answer should score the same.
- Do NOT penalize an answer for also citing chunk/page numbers.
- If the ACTUAL ANSWER captures the core fact(s) in EXPECTED ANSWER without
  contradicting them, mark PASS even if wording differs.
- If the ACTUAL ANSWER is missing the core fact, contradicts it, invents
  facts not supported by EXPECTED ANSWER, or evades the question, mark FAIL.
- If it's partially correct (gets some but not all required facts), mark PARTIAL.
- For questions marked as a false premise or requiring a refusal, judge
  whether the system correctly identified/handled that, per the EXPECTED ANSWER.

Respond with ONLY a JSON object, no other text:
{"verdict": "PASS" | "PARTIAL" | "FAIL", "reasoning": "<one short sentence>"}
"""


def judge_refusal(actual_answer):
    is_refusal = REFUSAL_STRING in actual_answer.lower()
    verdict = "PASS" if is_refusal else "FAIL"
    reasoning = (
        "Exact required refusal string found."
        if is_refusal
        else "Expected the exact refusal sentence; model answered or paraphrased instead."
    )
    return verdict, reasoning


def judge_open_ended(question, expected_answer, actual_answer):
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"EXPECTED ANSWER / CRITERIA: {expected_answer}\n\n"
        f"ACTUAL ANSWER: {actual_answer}"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # models sometimes wrap JSON in markdown fences despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[4:]

    try:
        parsed = json.loads(raw)
        verdict = str(parsed.get("verdict", "FAIL")).upper()
        reasoning = parsed.get("reasoning", "")
        if verdict not in ("PASS", "PARTIAL", "FAIL"):
            verdict = "FAIL"
            reasoning = f"Judge returned unrecognized verdict: {raw[:200]}"
    except (json.JSONDecodeError, AttributeError):
        verdict, reasoning = "FAIL", f"Judge output not parseable as JSON: {raw[:200]}"

    return verdict, reasoning