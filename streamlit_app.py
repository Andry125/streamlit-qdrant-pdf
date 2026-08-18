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

st.title("📚 PDF + Qdrant Cloud")

tab1, tab2 = st.tabs(["📤 Upload", "🔍 Search"])

with tab1:
    uploaded_file = st.file_uploader("Choisis un PDF", type="pdf")

    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)

        chunks = []
        page_numbers = []

        # Découper page par page en morceaux de 500 caractères
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                for i in range(0, len(text), 500):
                    chunk = text[i:i+500]
                    chunks.append(chunk)
                    page_numbers.append(page_num)

        vectors = model.encode(chunks)

        # Créer la collection si elle n'existe pas
        try:
            client.create_collection(
                collection_name="pdf_docs",
                vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
            )
        except Exception:
            st.info("La collection existe déjà, on continue.")

        # Insérer les points avec numéro de page
        client.upsert(
            collection_name="pdf_docs",
            points=[
                models.PointStruct(
                    id=i,
                    vector=vectors[i],
                    payload={"text": chunks[i], "page": page_numbers[i]}
                )
                for i in range(len(chunks))
            ]
        )

        st.success("✅ PDF indexé dans Qdrant Cloud avec numéros de page !")

with tab2:
    query = st.text_input("Entre ta requête (mot-clé ou phrase)")

    if query:
        query_vector = model.encode(query).tolist()  # convertir en liste
        try:
            results = client.search_points(
                collection_name="pdf_docs",
                query=query_vector,
                limit=5
            )

            st.write("Résultats :")
            for r in results:
                st.write(f"**Score:** {r.score:.4f} | **Page:** {r.payload.get('page', '?')}")
                st.write(r.payload["text"])
                st.write("---")
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {e}")
