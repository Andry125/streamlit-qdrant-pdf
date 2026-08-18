import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json

# Charger secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# --- Interface Streamlit ---
st.title("🔎 API Qdrant via Streamlit")

# Récupérer le paramètre GET ?q=motcle
query = st.query_params.get("q", "")

if query:
    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name="pdf_docs",
        query=query_vector,
        limit=5
    )

    results = [
        {
            "score": sp.score,
            "page": sp.payload.get("page", "?"),
            "text": sp.payload.get("text", "")
        }
        for sp in response.points
    ]

    st.json(results)
else:
    st.write("Ajoute ?q=motcle à l’URL pour tester.")
