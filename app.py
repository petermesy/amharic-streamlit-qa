
import streamlit as st
import os
import gdown
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_ID = '1689CqqDTTEG6b0N6-mw7hHD_IU6Cyq64'
POINTS_ID = '1VoBv3YRZR35FqSplnGUlwWh2fNKDoi3l'
CHUNKS_PATH = 'amharic-chunks-with-embeddings.jsonl'
POINTS_PATH = 'amharic-sentences-points.pkl'

def download_if_needed(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)

@st.cache_resource
def load_data():
    download_if_needed(CHUNKS_ID, CHUNKS_PATH)
    download_if_needed(POINTS_ID, POINTS_PATH)

    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    with open(POINTS_PATH, 'rb') as f:
        points = pickle.load(f)

    return chunks, np.array(points)

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def get_top_chunks(question_embedding, embeddings, chunks, top_k=3):
    similarities = np.dot(embeddings, question_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def main():
    st.title("Amharic QA System 🔍")
    chunks, embeddings = load_data()
    model = load_model()

    question = st.text_input("Ask a question in Amharic")
    if question:
        q_embed = model.encode(question)
        top_chunks = get_top_chunks(q_embed, embeddings, chunks)
        st.subheader("Top Relevant Chunks:")
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**{i}.** {chunk.get('text', '')[:400]}...")

if __name__ == "__main__":
    main()
