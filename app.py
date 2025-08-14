import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

import torch
import os

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model (no need to manually assign to device anymore)
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Path to the .pkl file
POINTS_FILE = "amharic_sentences_points.pkl"

# Load precomputed sentence embeddings
@st.cache_resource
def load_points():
    with open(POINTS_FILE, "rb") as f:
        return pickle.load(f)

points = load_points()

# Local similarity search using cosine similarity
def local_similarity_search(query, points, limit=5):
    query_vector = embedding_model.encode(query)
    vectors = np.array([point["vector"] for point in points])
    payloads = [point["payload"] for point in points]
    query_vector = np.array(query_vector)

    similarities = np.dot(vectors, query_vector) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    )

    top_indices = np.argsort(similarities)[::-1][:limit]
    results = [
        {"text": payloads[i]["text"], "score": float(similarities[i])}
        for i in top_indices
    ]
    return results


# Simple automatic summarization: concatenate top matches and return a short summary
def summarize_automatically(matches, query):
    combined_text = " ".join([match["text"] for match in matches])
    # Simple extractive summary: return the first 2-3 sentences
    sentences = combined_text.split("።")
    summary = "።".join(sentences[:3]).strip()
    if not summary.endswith("።"):
        summary += "።"
    return summary

# Streamlit UI
st.title("Amharic QA System")

query = st.text_input("የጥያቄዎትን ጽሑፍ ያስገቡ (Enter your Amharic question):")

if query:
    results = local_similarity_search(query, points, limit=5)
    summary = summarize_automatically(results, query)
    st.subheader("📝 አጭር መጠቃለያ")
    st.write(summary)
