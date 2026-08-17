import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import fitz  # PyMuPDF -> pip install pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()  # reads GROQ_API_KEY from your .env file into os.environ
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from groq import Groq  # pip install groq  (free API key: https://console.groq.com)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer ONLY using the context "
    "provided below. Do not use any outside knowledge, even if you know the "
    "answer. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "When you do answer, mention which chunk number(s) support your answer."
)


def load_and_chunk(file_path):
    """Ingestion + chunking, keeping page-level metadata for source references."""
    pdf_doc = fitz.open(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_chunks = []
    for page_num, page in enumerate(pdf_doc, start=1):
        text = page.get_text()
        if not text:
            continue
        for chunk_text in splitter.split_text(text):
            all_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={"source": os.path.basename(file_path), "page": page_num},
                )
            )
    pdf_doc.close()
    return all_chunks


def build_vector_store(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings_model)


def retrieve(file_path, query, k=3):
    """Full retrieval step: ingest -> chunk -> embed -> similarity search."""
    chunks = load_and_chunk(file_path)
    vector_store = build_vector_store(chunks)
    return vector_store.similarity_search(query, k=k)


def format_context(retrieved_docs):
    blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Chunk {i} | page {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def call_llm(user_prompt, system_prompt=SYSTEM_PROMPT):
    """The actual generation step: send instructions + context/question to an LLM."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content