import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from qdrant_client.http import models

# Charger secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

st.title("📚 Uploader PDF vers Qdrant Cloud")

uploaded_file = st.file_uploader("Choisis un PDF", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in reader.pages])

    st.write("Texte extrait (aperçu) :")
    st.write(text[:500])

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = model.encode(chunks)

    client.recreate_collection(
        collection_name="pdf_docs",
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )

    client.upsert(
        collection_name="pdf_docs",
        points=[
            models.PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
            for i in range(len(chunks))
        ]
    )

    st.success("✅ PDF indexé dans Qdrant Cloud !")
