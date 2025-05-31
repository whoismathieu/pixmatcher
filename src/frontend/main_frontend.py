""" Fichier gérant l'affichage général du site, avec un menu de navigation et un titre commun a toutes les pages. """

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

from streamlit_option_menu import option_menu

st.set_page_config(page_title="PixMatcher App", layout="wide")
st.markdown(
    """
    <style>
        h1 {
            color: black !important;
            font-weight: bold;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Welcome to the PixMatcher app")

# Menu de navigation dans la barre latérale
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Search via Image","Search via Text","Visualization", "About"],
        icons=["bi-image", "bi-chat-right-text","bar-chart", "info-circle"],
        menu_icon="list",
        default_index=0,
    )
# Charger et exécuter la page sélectionnée correct
if page == "Search via Image":
    import research
    research.main()  # Appelle la fonction main() correctement
elif page == "Search via Text":
    import clip_research
    clip_research.main()
elif page == "Visualization":
    import visualization
    visualization.main()
elif page == "About":
    import about
    about.main()

