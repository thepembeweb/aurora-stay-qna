
import os
from langchain_openai import ChatOpenAI

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available AuroraStay policy documents do not contain enough information "
    "to answer this question."
)

def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A vector store is required.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10.")
    results = vectorstore.similarity_search(query.strip(), k=k)
    return [
        f"Content:\n{result.page_content}\n\nMetadata:\n{result.metadata}"
        for result in results
    ]

def generate(query: str, retrieved_chunks: list[str], model):
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""
You are an AI assistant answering questions about AuroraStay hotel policies.

User Query:
{query}

Retrieved Policy Information:
{context}

Rules:
1. Use only facts explicitly supported by the retrieved AuroraStay policy information.
2. Answer the guest's specific question directly and concisely.
3. Include relevant conditions, exceptions, deadlines, fees, and property details.
4. Do not add outside knowledge, assumptions, or invented policy details.
5. Identify the source document from metadata when useful.
6. If evidence is insufficient, say: "{INSUFFICIENT_EVIDENCE_MESSAGE}"
7. If passages conflict, describe the conflict instead of choosing a value.

Answer:
"""
    return model.invoke(prompt)

def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key or not base_url:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_API_BASE must be configured.")
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
