""" Page à propos du siteweb, regroupant informations sur les technologies et sur les membres du groupe."""

import streamlit as st


def main():
    st.markdown(
        """
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
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("About the Project")

    st.markdown("""
    ## Overview
    This project, **L3E1**, was developed as part of the **Bachelor's degree in Computer Science at Université Paris 
    Cité**. PixMatcher implements a **Content-Based Image Retrieval (CBIR)** system, using descriptors extracted from 
    pre-trained Convolutional Neural Networks (CNN) to efficiently index and compare images. Additionally, a **Text-Based 
    Image Retrieval (TBIR)** feature has been implemented, allowing image searches based on text descriptions.

    ## Features
    - 🔍 **Image Similarity Search** using **MobileNetV3** or **CLIP** for feature extraction.  
    **MobileNetV3** is a lightweight convolutional neural network (CNN) optimized for mobile and embedded vision 
    applications. In this project, it is used to extract deep features from images, capturing essential visual 
    characteristics such as textures, shapes, and colors. These features serve as unique fingerprints that allow accurate 
    comparison between images.  
    **CLIP** (Contrastive Language-Image Pretraining), developed by OpenAI, is a powerful vision-language model that 
    learns visual concepts from natural language supervision. In this project, CLIP enables image retrieval based on 
    textual queries, allowing users to search for images using descriptive text rather than another image.

    - ⚡ **Fast Indexing & Retrieval** powered by **FAISS** (Facebook AI Similarity Search). 
    FAISS is a high-performance library developed by Facebook AI for efficient similarity search on large-scale 
    datasets. It enables rapid nearest-neighbor searches in high-dimensional feature spaces, making it ideal for finding 
    visually similar images in a database with minimal latency.
    
    - 🏞️ **Image Set** by Tiny ImageNet or Open Image.
    **Tiny ImageNet** is a smaller version of the ImageNet dataset, containing 200 object classes with 500 images each. 
    It is commonly used for benchmarking image recognition and retrieval tasks. By leveraging this dataset, the project 
    ensures a diverse and challenging image retrieval environment.
    **Open Images** is a large-scale dataset collected by Google, composed of real-world images sourced from the web. 
    Compared to Tiny ImageNet, it offers greater visual diversity and significantly higher image quality, making it 
    ideal for more realistic image retrieval tasks.
    
    - 📊 **Data Visualization** with **t-SNE** for feature space exploration.
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that helps visualize high-dimensional data in 2D or 3D. In this project, it is used to display the feature embeddings of images, allowing users to understand how similar images are grouped in the feature space.
    - 🖥 **User-Friendly Interface** developed with **Streamlit**.
    Streamlit is a Python framework for building interactive web applications with minimal coding effort. It provides an intuitive and visually appealing user interface for performing image searches, viewing retrieval results, and interacting with the system seamlessly.

    ## Team
    This project was developed by a team of four students:
    - **👩‍💻Nassilya Belguedj**
    - **👨‍💻Aaron Aidoudi**
    - **👨‍💻Mathieu Moustache**
    - **👩‍💻Julie Colliere**
    
    Supervised by **👨‍🏫Mr. Camille Kurtz**
    
    Each team member contributed to different aspects of the project, including feature extraction, indexing, visualization, interface development, and documentation

    ## Contact
    For any inquiries or further information about the project, feel free to reach out to us!
    """)


if __name__ == "__main__":
    main()
