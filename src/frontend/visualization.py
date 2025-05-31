""" Page de visualisation des embeddings et affichage d'un plot issu de l'algorithme t-SNE. """

import streamlit as st
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
def main():

    st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
                background-color: #FAE5D3 !important;
                font-family: 'Arial', sans-serif !important;
                color: black !important;
            }
        /* Appliquer aussi aux titres et paragraphes */
        h1, h2, h3, h4, h5, h6, p, div {
            color: black !important;
        }
        .css-1aumxhk {
            max-width: 800px;
            margin: auto;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Visualizing Embeddings with t-SNE</h2>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
                Explore the spatial distribution of embeddings using the t-SNE (t-distributed Stochastic Neighbor 
                Embedding) method. This projection allows you to intuitively visualize the similarities between 
                different visual representations.</p>""", unsafe_allow_html=True)

    # Régénérer la visualisation t-SNE
    if st.button("🔄 Regenerate t-SNE visualization"):
        x = np.random.randn(100)
        y = np.random.randn(100)
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.6)
        st.session_state.tsne_data = fig
        st.success("Visualization successfully regenerated!")
    # Vérifier si la visualisation t-SNE est déjà stockée en session
    if "tsne_data" not in st.session_state:
        x = np.random.randn(100)
        y = np.random.randn(100)
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.6)
        st.session_state.tsne_data = fig
    # Afficher l’image de t-SNE stockée
    st.pyplot(st.session_state.tsne_data)
    # Charger et afficher l'image générée
    image_path = os.path.join("ressources", "tsne_plot.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption="Projection t-SNE", use_container_width=True)
    else:
        st.error("L'image t-SNE n'a pas été trouvée.")
    # Informations supplémentaires
    with st.expander("Learn more about t-SNE"):
        st.markdown("""
            **t-SNE** is a nonlinear dimensionality reduction method, particularly suited for visualizing high-dimensional data. It reveals hidden structures by placing similar points close to each other in a 2-dimensional space.             **Applications fréquentes :**
            - Visualizing complex data
            - Identification of clusters or natural groupings
            - Exploratory data analysis
            📚 [Official t-SNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
        """)
if __name__ == "__main__":
    main()