
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")
st.set_page_config(page_title="AuroraStay Policy Assistant", page_icon="🏨", layout="wide")

with st.sidebar:
    st.header("Model parameters")
    k = st.slider("Chunks to retrieve", 1, 10, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.1)
    max_tokens = st.slider("Maximum answer tokens", 64, 2048, 512, 64)

st.title("🏨 AuroraStay Policy Assistant")
st.write("Ask about AuroraStay booking, cancellation, charges, rooms, conduct, or accessibility policies.")
response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True,
)
query = st.text_input(
    "Your question",
    placeholder="e.g., What fees or restrictions apply when a guest brings a pet?",
)

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a hotel-policy question.")
    else:
        endpoint = "/v1/answer" if response_type == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {
            "query": query.strip(),
            "k": k,
            "model_name": "gpt-4o-mini",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            with st.spinner("Searching AuroraStay policies..."):
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=120)
            if response.ok:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                if response_type == "Answer + Relevant Chunks":
                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Chunk {i}"):
                            st.text(chunk)
            else:
                st.error(f"Backend returned HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as error:
            st.error(f"Unable to connect to the RAG API at {BACKEND_URL}: {error}")
