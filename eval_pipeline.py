import time
from rag_utils import client, SYSTEM_PROMPT
from retrieval_prompt_setup import setup_retrieval_and_prompt

MODEL_NAME = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was deprecated 08/16/26; official replacement


def run_pipeline_instrumented(file_path, query, k=3, system_prompt=None):
    """Runs retrieval + generation, returns the answer plus timing/token metrics.

    system_prompt: pass an alternate system prompt (e.g. SYSTEM_PROMPT_V2 from
    few_shot_fix.py) to evaluate a prompt variant without touching rag_utils.py.
    """
    active_system_prompt = system_prompt or SYSTEM_PROMPT

    t0 = time.perf_counter()
    retrieved_docs, messages = setup_retrieval_and_prompt(file_path, query, k=k)
    t1 = time.perf_counter()
    retrieval_latency = t1 - t0

    # setup_retrieval_and_prompt always builds messages with the ORIGINAL
    # SYSTEM_PROMPT baked in -- override it here if a variant was requested.
    messages[0]["content"] = active_system_prompt

    t2 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    t3 = time.perf_counter()
    generation_latency = t3 - t2

    answer = response.choices[0].message.content
    usage = response.usage  # groq's response.usage: prompt_tokens, completion_tokens, total_tokens

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "retrieval_latency_s": retrieval_latency,
        "generation_latency_s": generation_latency,
        "total_latency_s": t3 - t0,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }