import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration des chemins vers les fichiers du modèle ---
MODEL_DIR = "models"
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
AHP_WEIGHTS_PATH = os.path.join(MODEL_DIR, "ahp_weights.pt")
MODEL_PATH = os.path.join(MODEL_DIR, "ahp_transformer_model.pth")

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="NTDs Diagnostic Assistant",
    page_icon="🩺", # Icône pour l'onglet du navigateur
    layout="wide", # Utilise toute la largeur disponible de l'écran
    initial_sidebar_state="expanded" # La barre latérale est ouverte par défaut
)

# --- CSS Personnalisé pour un design professionnel et esthétique ---
# Injecte du CSS pour personnaliser l'apparence des éléments Streamlit
st.markdown("""
    <style>
    /* Styles généraux du corps de l'application */
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; /* Police moderne et lisible */
        color: #333; /* Couleur de texte foncée */
        background-color: #f0f2f6; /* Arrière-plan léger et doux */
    }
    /* Conteneur principal de l'application */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Conteneur de blocs pour un espacement agréable */
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px; /* Limite la largeur du contenu principal */
        margin: auto; /* Centre le contenu */
    }
    /* Styles des titres */
    h1 {
        color: #004d40; /* Vert sarcelle foncé */
        font-size: 2.5em;
        margin-bottom: 0.5em;
        text-align: center;
        border-bottom: 2px solid #e0e0e0; /* Ligne de séparation subtile */
        padding-bottom: 10px;
    }
    h2 {
        color: #00695c; /* Vert sarcelle moyen */
        font-size: 2em;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    h3 {
        color: #00796b; /* Vert sarcelle clair */
        font-size: 1.5em;
        margin-top: 1em;
        margin-bottom: 0.6em;
    }
    /* Styles des boutons */
    .stButton>button {
        background-color: #00796b; /* Vert sarcelle */
        color: white;
        border-radius: 8px; /* Bords arrondis */
        padding: 0.6em 1.5em;
        font-size: 1.1em;
        border: none;
        transition: all 0.2s ease-in-out; /* Transition douce au survol */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Ombre légère */
    }
    .stButton>button:hover {
        background-color: #004d40; /* Vert sarcelle plus foncé au survol */
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); /* Ombre plus prononcée */
        transform: translateY(-2px); /* Effet de léger soulèvement */
    }
    /* Styles des Selectbox et Text Input */
    .stSelectbox>div>div, .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 0.5em 1em;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05); /* Ombre interne subtile */
    }
    .stSelectbox>div>div:focus, .stTextInput>div>div>input:focus {
        border-color: #00796b; /* Bordure sarcelle au focus */
        box-shadow: 0 0 0 0.2rem rgba(0, 121, 107, 0.25); /* Ombre de focus */
    }
    /* Styles des messages (Info, Warning, Success) */
    .stAlert {
        border-radius: 8px;
        padding: 1em;
        margin-top: 1em;
    }
    .stAlert.info {
        background-color: #e0f7fa; /* Cyan très clair */
        color: #006064; /* Cyan foncé */
        border-left: 5px solid #00bcd4; /* Bordure cyan */
    }
    .stAlert.warning {
        background-color: #fff3e0; /* Orange très clair */
        color: #e65100; /* Orange foncé */
        border-left: 5px solid #ff9800; /* Bordure orange */
    }
    .stAlert.success {
        background-color: #e8f5e9; /* Vert très clair */
        color: #2e7d32; /* Vert foncé */
        border-left: 5px solid #4caf50; /* Bordure verte */
    }
    /* Styles des Expanders */
    .streamlit-expanderHeader {
        background-color: #e0f2f7; /* Bleu très clair */
        color: #004d40;
        border-radius: 8px;
        padding: 0.8em 1em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
        border: 1px solid #b2ebf2;
    }
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-top: none;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        padding: 1em;
    }
    /* Styles de la barre latérale */
    .css-1lcbmhc, .css-1d3f8gv { /* Classes Streamlit pour la barre latérale */
        background-color: #e0f2f7; /* Bleu clair pour la barre latérale */
        padding: 1rem;
    }
    .css-1lcbmhc h1, .css-1d3f8gv h1 {
        color: #004d40;
    }
    </style>
    """, unsafe_allow_html=True) # Permet à Streamlit d'interpréter le HTML/CSS

# ===============================
# Définition du Modèle (doit être identique au script d'entraînement)
# ===============================
class AHPTransformer(nn.Module):
    def __init__(self, input_dim, n_classes, ahp_weights):
        super().__init__()
        self.att = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, n_classes)
        # Enregistrer ahp_weights comme un buffer pour qu'il soit sauvegardé avec le modèle
        # mais ne soit pas un paramètre entraînable.
        self.register_buffer('ahp_weights', ahp_weights)

    def forward(self, x):
        scores = self.att(x)
        scores = scores * self.ahp_weights
        w = torch.softmax(scores, dim=1)
        x = w * x
        return self.fc(x), w

# ===============================
# Chargement du Modèle et des Préprocesseurs
# Utilisation de st.cache_resource pour charger ces éléments une seule fois
# ===============================
@st.cache_resource
def load_model_and_preprocessors():
    """
    Charge le modèle entraîné, l'encodeur d'étiquettes et les noms de colonnes
    pour le prétraitement.
    """
    try:
        # Charger LabelEncoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        # Charger les noms des colonnes des caractéristiques
        with open(FEATURE_COLUMNS_PATH, "rb") as f:
            feature_columns = pickle.load(f)

        # Charger les poids AHP
        ahp_weights = torch.load(AHP_WEIGHTS_PATH)

        # Déterminer INPUT_DIM et N_CLASSES à partir des composants chargés
        input_dim = len(feature_columns)
        n_classes = len(label_encoder.classes_)

        # Initialiser le modèle et charger son état
        model = AHPTransformer(input_dim, n_classes, ahp_weights)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Mettre le modèle en mode évaluation (désactive dropout, batchnorm, etc.)
        print("Modèle et préprocesseurs chargés avec succès.")
        return model, label_encoder, feature_columns
    except FileNotFoundError as e:
        st.error(f"Erreur: Fichiers de modèle ou de préprocesseur introuvables. "
                 f"Veuillez d'abord exécuter `train_and_save_model.py`. Détails: {e}")
        st.stop() # Arrête l'application si les fichiers essentiels sont manquants
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement du modèle ou des préprocesseurs: {e}")
        st.exception(e) # Affiche la trace complète de l'erreur pour le débogage
        st.stop()

# Charger le modèle et les préprocesseurs au démarrage de l'application
model, label_encoder, feature_columns = load_model_and_preprocessors()

# ===============================
# Fonction de prétraitement pour l'inférence
# ===============================
def preprocess_input(input_data: dict, feature_columns: list) -> torch.Tensor:
    """
    Prétraite les données d'entrée de l'utilisateur pour les rendre compatibles
    avec le modèle.
    """
    # Créer un DataFrame à partir du dictionnaire d'entrée
    input_df = pd.DataFrame([input_data])

    # Appliquer l'encodage one-hot
    processed_df = pd.get_dummies(input_df)

    # Réindexer pour s'assurer que toutes les colonnes d'entraînement sont présentes
    # et dans le bon ordre. Les colonnes manquantes seront remplies avec 0.
    # Les colonnes supplémentaires (si des catégories inattendues apparaissent) seront supprimées.
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    # Convertir en tenseur PyTorch
    return torch.tensor(processed_df.values, dtype=torch.float32)

# ===============================
# Gestion de l'état de la session pour la connexion
# ===============================
if "logged" not in st.session_state:
    st.session_state.logged = False

# ===============================
# Barre latérale pour la Navigation et les Informations
# ===============================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Streamlit_logo.png/500px-Streamlit_logo.png", width=150) # Logo Streamlit comme placeholder
    st.header("Navigation")
    if st.session_state.logged:
        #st.page_link("main.py", label="🩺 Assistant de Diagnostic", icon="🏠")
        st.page_link("https://www.who.int/news-room/fact-sheets/detail/neglected-tropical-diseases", label="🌐 À propos des MTN", icon="ℹ️")
        st.page_link("https://github.com/votre_utilisateur/votre_repo", label="💻 Code Source", icon="⚙️") # Remplacez par votre vrai dépôt GitHub
        st.markdown("---")
        if st.button("Déconnexion", help="Cliquez pour vous déconnecter du système."):
            st.session_state.logged = False
            st.rerun() # Redémarre l'application pour afficher la page de connexion
    else:
        st.info("Veuillez vous connecter pour accéder à l'outil de diagnostic.")
        st.page_link("https://www.who.int/news-room/fact-sheets/detail/neglected-tropical-diseases", label="🌐 À propos des MTN", icon="ℹ️")
        st.markdown("---")
        st.caption("Version 1.0")

# ===============================
# PAGE 1 — Portail de Connexion
# ===============================
if not st.session_state.logged:
    st.title("🔐 Portail d'Accès Clinique")
    st.markdown("---")
    st.write("Bienvenue sur le portail d'accès clinique. Veuillez vous connecter pour utiliser l'assistant de diagnostic des Maladies Tropicales Négligées (MTN).")

    # Utilisation de st.form pour regrouper les éléments de connexion
    with st.form("login_form"):
        user = st.text_input("Nom d'utilisateur", help="Entrez votre nom d'utilisateur clinique.")
        pwd = st.text_input("Mot de passe", type="password", help="Entrez votre mot de passe sécurisé.")
        login_button = st.form_submit_button("Se connecter")

        if login_button:
            # Authentification factice pour la démonstration
            if user == "clinician" and pwd == "securepass": # Remplacez par une logique d'authentification réelle
                st.session_state.logged = True
                st.success("Connexion réussie ! Redirection vers l'assistant de diagnostic.")
                st.rerun() # Redémarre l'application pour afficher la page de diagnostic
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
                st.session_state.logged = False

# ===============================
# PAGE 2 — Assistant de Diagnostic
# ===============================
else:
    st.title("🩺 Assistant de Diagnostic des MTN (AHP-Attention)")
    st.markdown("---")
    st.write("Veuillez entrer les informations cliniques du patient ci-dessous pour générer un diagnostic potentiel.")

    # Utilisation de st.form pour regrouper les entrées du formulaire de diagnostic
    with st.form("diagnostic_form", clear_on_submit=False): # clear_on_submit=False pour garder les valeurs après soumission
        st.subheader("Informations sur les Lésions")
        col1, col2 = st.columns(2) # Crée deux colonnes pour une meilleure disposition des entrées

        with col1:
            location = st.selectbox("Localisation de la lésion", ["ankle","leg","foot","face","trunk"], help="Où la lésion est-elle située sur le corps du patient?")
            lesion_number = st.selectbox("Nombre de lésions", ["single","few","multiple"], help="Combien de lésions sont observées chez le patient?")
            lesion_type = st.selectbox("Type de lésion", ["ulcer","nodule","swelling","itchy"], help="Quelle est la nature principale de la lésion (ex: ulcère, nodule)?")
        with col2:
            base = st.selectbox("Base de la lésion", ["clean","crusted","necrotic"], help="Décrivez l'aspect de la base de la lésion (ex: propre, croûteuse, nécrotique).")
            bilateral = st.selectbox("Bilatéral", ["yes","no"], help="Les lésions sont-elles présentes des deux côtés du corps (oui/non)?")
            pain = st.selectbox("Douleur", ["yes","no"], help="Le patient ressent-il de la douleur associée à la lésion (oui/non)?")
            edema = st.selectbox("Œdème", ["yes","no"], help="Y a-t-il un œdème (gonflement) présent autour de la lésion (oui/non)?")
            varicose_veins = st.selectbox("Varices", ["yes","no"], help="Le patient présente-t-il des varices (oui/non)?")

        submit_button = st.form_submit_button("Générer le Diagnostic")

    if submit_button:
        # Collecter les entrées de l'utilisateur
        inputs = {
            "location": location,
            "lesion_number": lesion_number,
            "lesion_type": lesion_type,
            "base": base,
            "bilateral": bilateral,
            "pain": pain,
            "edema": edema,
            "varicose_veins": varicose_veins
        }

        with st.spinner("Analyse des données cliniques et génération du diagnostic..."):
            try:
                # Prétraiter les entrées pour le modèle
                input_tensor = preprocess_input(inputs, feature_columns)

                # Effectuer la prédiction avec le modèle
                with torch.no_grad(): # Désactiver le calcul des gradients pour l'inférence
                    logits, _ = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0] # Obtenir les probabilités
                    predicted_class_idx = torch.argmax(probabilities).item() # Index de la classe prédite
                    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0] # Nom de la classe

                st.success("Diagnostic généré avec succès !")

                st.subheader("📝 Résumé Clinique")
                st.info(
                    f"Le patient présente **{inputs['lesion_number']}** "
                    f"lésion(s) de type **{inputs['lesion_type']}** "
                    f"située(s) à **{inputs['location']}**, "
                    f"avec une base **{inputs['base']}**. "
                    f"Présence bilatérale: **{inputs['bilateral']}**, "
                    f"douleur: **{inputs['pain']}**, "
                    f"œdème: **{inputs['edema']}**, "
                    f"varices: **{inputs['varicose_veins']}**."
                )

                st.subheader("💡 Catégorie Clinique Prédite")
                st.markdown(f"La catégorie la plus probable est : <span style='font-size: 1.5em; font-weight: bold; color: #00796b;'>{predicted_label}</span>", unsafe_allow_html=True)

                st.subheader("📊 Probabilités de Diagnostic")
                # Afficher les probabilités pour toutes les classes dans un tableau
                prob_df = pd.DataFrame({
                    "Catégorie": label_encoder.classes_,
                    "Probabilité": probabilities.numpy()
                }).sort_values(by="Probabilité", ascending=False) # Trier par probabilité décroissante
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

                st.caption("⚠️ Ceci est un outil d'aide à la décision uniquement et ne constitue pas un diagnostic médical. Toujours consulter un professionnel de la santé qualifié pour un diagnostic définitif.")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la prédiction: {e}")
                st.exception(e) # Affiche la trace complète de l'erreur pour le débogage

# Ajouter un pied de page
st.markdown("---")
st.markdown("Développé avec ❤️ pour l'aide au diagnostic des MTN. © 2026")