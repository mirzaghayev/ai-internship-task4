import time
from rag_utils_v2 import client, SYSTEM_PROMPT_V2, retrieve_v2, format_context

MODEL_NAME = "openai/gpt-oss-20b"  # llama-3.1-8b-instant deprecated 08/16/26


def run_pipeline_v2(file_path, query, k=5, system_prompt=None):
    active_system_prompt = system_prompt or SYSTEM_PROMPT_V2

    t0 = time.perf_counter()
    retrieved_docs = retrieve_v2(file_path, query, k=k)
    t1 = time.perf_counter()
    retrieval_latency = t1 - t0

    context = format_context(retrieved_docs)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    messages = [
        {"role": "system", "content": active_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t2 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    t3 = time.perf_counter()
    generation_latency = t3 - t2

    answer = response.choices[0].message.content
    usage = response.usage

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