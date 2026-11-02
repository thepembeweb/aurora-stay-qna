
from flask import Flask, jsonify, request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_utils import rag, retrieve

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"

rag_api = Flask("AuroraStay RAG API")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"Vectors stored: {vectorstore.index.ntotal}")

def parse_payload(require_model_parameters=False):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    k = data.get("k", 2)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10.")

    parsed = {"query": query.strip(), "k": k}
    if require_model_parameters:
        model_name = data.get("model_name", "gpt-4o-mini")
        temperature = data.get("temperature", 0.0)
        top_p = data.get("top_p", 1.0)
        max_tokens = data.get("max_tokens", 512)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 64 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be an integer between 64 and 2048.")
        parsed.update(
            model_name=model_name.strip(),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=max_tokens,
        )
    return parsed

@rag_api.errorhandler(ValueError)
def handle_invalid_request(error):
    return jsonify({"error": str(error)}), 400

@rag_api.get("/")
def home():
    return jsonify({"service": "AuroraStay RAG API", "status": "ready"})

@rag_api.get("/health")
def health():
    return jsonify({"status": "healthy", "vectors": int(vectorstore.index.ntotal)})

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    parameters = parse_payload()
    chunks = retrieve(vectorstore=vectorstore, **parameters)
    return jsonify({**parameters, "relevant_chunks": chunks})

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    parameters = parse_payload(require_model_parameters=True)
    generated_answer, chunks = rag(vectorstore=vectorstore, **parameters)
    return jsonify({**parameters, "answer": generated_answer, "relevant_chunks": chunks})

@rag_api.post("/v1/answer")
def answer():
    parameters = parse_payload(require_model_parameters=True)
    generated_answer, _ = rag(vectorstore=vectorstore, **parameters)
    return jsonify({"answer": generated_answer})

if __name__ == "__main__":
    rag_api.run(host="0.0.0.0", port=7860, debug=False)
