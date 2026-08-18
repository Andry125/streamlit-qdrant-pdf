import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json

# Charger secrets depuis Streamlit Cloud (Settings → Secrets)
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Initialiser Qdrant et le modèle
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# --- Interface Streamlit ---
st.set_page_config(page_title="Qdrant API", page_icon="🔎")
st.title("🔎 API Qdrant via Streamlit")

# Récupérer paramètres GET
query = st.query_params.get("q", "")
fmt = st.query_params.get("format", "")

if query:
    # Encoder le texte en vecteur
    query_vector = model.encode(query).tolist()

    # Interroger Qdrant
    response = client.query_points(
        collection_name="pdf_docs",
        query=query_vector,
        limit=5
    )

    # Formater les résultats
    results = [
        {
            "score": sp.score,
            "page": sp.payload.get("page", "?"),
            "text": sp.payload.get("text", "")
        }
        for sp in response.points
    ]

    # Retour JSON brut si format=json
    if fmt == "json":
        st.write(json.dumps(results))   # JSON pur (pas de redirection)
    else:
        st.json(results)                # Affichage interactif Streamlit
else:
    st.write("Ajoute ?q=motcle à l’URL pour tester, ex: ?q=poker&format=json")
