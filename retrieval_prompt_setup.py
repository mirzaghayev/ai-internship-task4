from rag_utils import retrieve, format_context, SYSTEM_PROMPT


def setup_retrieval_and_prompt(file_path, query, k=3):
    """
    Runs retrieval and assembles the final prompt.
    Returns both the retrieved chunks (for source tracking downstream)
    and the messages ready to send to an LLM, with instructions kept
    clearly separate from the retrieved context.
    """
    retrieved_docs = retrieve(file_path, query, k=k)
    context = format_context(retrieved_docs)

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  # instructions
        {"role": "user", "content": user_prompt},       # context + question
    ]

    return retrieved_docs, messages


# yoxlama bolumu:
if __name__ == "__main__":
    pdf_path = "test_file.pdf"
    query = "Who is Gregor Samsa?"

    print("Setting up retrieval and formatting prompt...")
    retrieved_docs, messages = setup_retrieval_and_prompt(pdf_path, query)

    print("\n--- SYSTEM (INSTRUCTIONS) ---")
    print(messages[0]["content"])

    print("\n--- USER (CONTEXT + QUESTION) ---")
    print(messages[1]["content"])