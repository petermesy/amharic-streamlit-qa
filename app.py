import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import torch
import os
import gdown

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

# Download the .pkl file from Google Drive
POINTS_FILE = "amharic-sentences-points.pkl"
GDRIVE_FILE_ID = "1VoBv3YRZR35FqSplnGUlwWh2fNKDoi3l"

@st.cache_resource
def download_and_load_points():
    if not os.path.exists(POINTS_FILE):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", POINTS_FILE, quiet=False)
    with open(POINTS_FILE, "rb") as f:
        return pickle.load(f)

points = download_and_load_points()
