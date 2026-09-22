import os
from functools import lru_cache
from . import *

import faiss
from groq import Groq
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
load_dotenv()


DOCUMENTS = [
    """
    Retrieval Augmented Generation (RAG) is a technique where an AI system
    retrieves relevant information from an external knowledge base before
    generating an answer.
    """,
    """
    LangChain is a framework for developing applications powered by language
    models. It provides components for prompts, models, retrievers, tools,
    agents, and document processing.
    """,
    """
    LangGraph is a framework for building stateful agent workflows.
    It allows developers to create graphs containing nodes, edges,
    conditional routing, loops, and tool calls.
    """,
    """
    FastAPI is a Python framework for building APIs. It is commonly used
    to expose machine learning and AI applications as web services.
    """,
    """
    Vector databases store vector representations of data and allow
    similarity searches. They are commonly used in RAG systems to retrieve
    information relevant to a user's query.
    
    DASARADAHA is a good boy who live in hyderabad and curently had an circumcision operation
    for the foreskin called circumcision. he is still recovering from the pain.
    """,
]

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def get_faiss_index():
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(
        DOCUMENTS,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_documents(query: str, k: int = 3):
    embedding_model = get_embedding_model()
    index = get_faiss_index()

    k = min(k, len(DOCUMENTS))

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(DOCUMENTS):
            results.append(DOCUMENTS[i])

    return results


def build_context(query: str, k: int = 3) -> str:
    retrieved_docs = retrieve_documents(query, k=k)
    return "\n\n".join(retrieved_docs)


def rag_answer(question: str) -> str:
    context = build_context(question)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the provided context.

If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}

Question:
{question}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


def direct_llm_answer(message: str) -> str:
    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
    )

    return response.choices[0].message.content
