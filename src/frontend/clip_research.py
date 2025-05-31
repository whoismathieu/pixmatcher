""" Page de recherche TBIR. Après une présentation du TBIR et un choix du dataset, l'utilisateur transmet sa text query
et la recherche s'effectue. """

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from src.clip_similarity_search import (oi_find_similar_images, oi_get_image_path, ti_find_similar_images,
                                        ti_get_image_path)


def main():
    st.markdown(
        """
        <style>
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #FAE5D3 !important;
                font-family: 'Arial', sans-serif !important;
                color: black !important;
            }
        .title, .subtitle, .description {
            color: black !important;
            font-weight: bold;
        }
        .stButton>button {
            color: white;
        }
        #MainMenu {
        visibility: hidden;
        }
        .header {
            background-color: #E8C3A6;
            padding: 20px;
            text-align: left;
            font-size: 24px;
            font-weight: bold;
            color: black;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .steps-container {
            text-align: center;
            padding: 20px;
        }
        .step {
            font-size: 18px;
            font-weight: bold;
            margin: 10px;
            color: #D2691E;
            display: inline-block;

        }
        .step-number {
            font-size: 24px;
            font-weight: bold;
            color: #FF8C00;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] > label > div {
            color: black !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }    

        </style>
        """, unsafe_allow_html=True)

    # Titre de la page
    st.markdown('<h2 class="subtitle">The AI that interprets textual content</h2>', unsafe_allow_html=True)

    # Description
    st.markdown('<h4 class="subtitle">📝 Text-Based Image Retrieval (TBIR)</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
                Text-Based Image Retrieval (TBIR) enables users to find relevant images based on textual descriptions. 
                Unlike traditional keyword-based searches, this approach leverages advanced AI models to understand the 
                meaning behind your words and match them with visually similar images.</p>
                """, unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">💡 How Does It Work?</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
                • Enter a short description of the image you are looking for.<br>
                • The AI interprets your text and converts it into a feature representation.<br>
                • The system searches through the image database to find the closest visual matches.<br> 
                """, unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">🔍 Why Use TBIR?</h4>', unsafe_allow_html=True)
    st.markdown("""<p style="color: black; font-size: 16px;">
                • No need to have an example image—just describe what you want to find.<br>
                • AI-powered understanding of concepts, objects, and scenes.<br>
                • Ideal for discovering images that match an idea, a mood, or a specific object. <br></p>
                """, unsafe_allow_html=True)

    # Étapes
    st.markdown("""
            <div class="steps-container">
                <span class="step"><span class="step-number">Step 1</span><br>Upload your image</span> 
                <span class="step"><span class="step-number">➝</span><br></span>
                <span class="step"><span class="step-number">Step 2</span><br>AI looks for similarities </span>
                <span class="step"><span class="step-number">➝</span><br></span>
                <span class="step"><span class="step-number">Step 3</span><br>Discover the assimilated images</span>
            </div>
        """, unsafe_allow_html=True)

    # Choix du dataset
    selected_dataset = st.radio("", ("Open Images", "Tiny ImageNet"), index=0, horizontal=True, )
    st.markdown(f'<h5 class="subtitle">📁 Perform a search on {selected_dataset}</h4>', unsafe_allow_html=True)

    # Recherche
    '''
    query = st.text_input("", placeholder="Enter description here")
    if st.button("Research"):
        try:
            top_indices = ti_find_similar_images(query)
            st.subheader("Assimilated images")
            cols = st.columns(5)  # Affichage des images en ligne

            for i, idx in enumerate(top_indices):
                img_path = ti_get_image_path(idx)
                image = Image.open(img_path)
                with cols[i % 5]:
                    st.image(image, caption=f"Image {i + 1}", use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")
    '''

    query = st.text_input("", placeholder="Enter description here")
    if st.button("Research"):
        try:
            if selected_dataset == "Open Images":
                top_indices = oi_find_similar_images(query,100)
                st.subheader("Assimilated images")
                cols = st.columns(5)  # Affichage en 5 colonnes

                for i, idx in enumerate(top_indices):
                    img_url = oi_get_image_path(idx)

                    if img_url:  # Charger l'image depuis une URL AWS
                        response = requests.get(img_url)
                        img = Image.open(BytesIO(response.content)).convert('RGB')

                        with cols[i % 5]:
                            st.image(img, caption=f"Image {i + 1}", use_container_width=True)
                    else:
                        st.error(f"Impossible de récupérer l'image pour l'index {idx}")

            if selected_dataset == "Tiny ImageNet":
                top_indices = ti_find_similar_images(query)
                st.subheader("Assimilated images")
                cols = st.columns(5)  # Affichage des images en ligne

                for i, idx in enumerate(top_indices):
                    img_path = ti_get_image_path(idx)
                    image = Image.open(img_path)
                    with cols[i % 5]:
                        st.image(image, caption=f"Image {i + 1}", use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")


if __name__ == "__main__":
    main()
