importateur élu par un ruisseau comme st
de qdrant_client importateur Client Qdrant
de transformateurs_de phrases importateur Transformateur de phrases
de pypdf importateur Lecteur PDF
de qdrant_client.http mode importateur

# Secrets du chargeur
QDRANT_URL = st.secrets["URL_QDRANT"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

client = Client Qdrant(url=QDRANT_URL, api_key=QDRANT_API_KEY)
modèle = Transformateur de phrases("distiluse-base-multilingual-cased-v1")

st.titre("📚 Téléchargeur PDF vers Qdrant Cloud")

fichier_téléchargé = st.téléchargeur_fichier("Choisis un PDF", type="pdf")

si fichier_téléchargé est pas Aucun :
    lecteur = Lecteur PDF(fichier_téléchargé)
    texte = "".rejoindre([page.extraire_texte() pour page dans lecteur.pages])

    st.écrire("Texte extrait (aperçu) :")
    st.écrire(texte[:500])

    morceaux = [texte[i:i+500] verser je danse gamme(0, len(texte), 500)]
    vecteurs = modèle.encodeur(morceaux)

    client.collection_recréer(
        nom_collection="pdf_docs",
        vecteurs_config=modèles.Paramètres vectoriels(taille=len(vecteurs[0]), distance=modèles.Distance.COSINUS),
    )

    client.insérer(
        nom_collection="pdf_docs",
        points=[
            modèles.Structure point(id=i, vecteur=vecteurs[i], charge utile={"texte": morceaux[i]})
            pour je dans gamme(len(morceaux))
        ]
    )

    st.succès("✅ PDF indexé dans Qdrant Cloud !")
