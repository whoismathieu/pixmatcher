# PixMatcher : L3E1 project for content-based image retrieval and indexing of an image database
This project implements a **Content-Based Image Retrieval (CBIR)** system, using descriptors extracted from pre-trained
**Convolutional Neural Networks (CNN)** to efficiently index and compare images. Additionally, a **Text-Based Image 
Retrieval (TBIR)** feature has been implemented, allowing image searches based on text descriptions.\\

Basically, for CBIR, the system converts query images and database images into high-dimensional feature vectors using a 
CNN, [MobileNetV3](#mobilenetv3). These vectors capture visual patterns such as shapes, textures, and colors. The system 
then compares the image vectors using [FAISS](#faiss), a library designed for fast similarity search.\
For TBIR, a model called [CLIP](#clip---vit-b32) (Contrastive Language–Image Pretraining) is used to map images and text 
into a shared vector space, enabling cross-modal search, retrieving relevant images from text queries.\
For both search methods, we use two datasets (at the user's discretion) : [Tiny ImageNet](#tiny-imagenet) and 
[Open Image](#open-images).


## Project modules
- **Image Preprocessing** : Via *image_preprocessing.py*, the module prepares images for input into a neural network.
- **Feature Extraction** : Via *feature_extractor.py*, uses MobileNetV3 and its large weights to extract feature vectors. Also, via *tinyimagenet_mobilenetv3_feature_extractor.py*, MobileNetV3 is also used to extract feature vectors from the dataset images.
- **Similarity Search** : Via *similarity_search.py*, the search is done with FAISS to quickly search images and the cosine distance for similar categories.

- **Feature Extraction for TBIR** : Via *tinyimagenet_clip_feature_extractor.py*, uses CLIP and its ViT-B/32 model to extract vectors from the dataset.
- **Similarity Search for TBIR** : In *clip_similarity_search.py*, the module converts the text into a feature vector to compare it to those in the dataset.

- **Front-end**: In */frontend*, use Streamlit for the interface.


## Architecture du Projet
```
PixMatcher/
│
├── requirements.txt            # List of dependencies
├── README.txt                  # Project documentation
│
├── src/                        # Main application source code folder
│   ├── __init__.py             # Marks the folder as a Python package
│   │
│   ├── image_preprocessing.py  # Image preprocessing module
│   ├── feature_extractor.py    # Feature extraction module, with MobileNetV3
│   ├── similarity_search.py    # Similar image search module, with FAISS and in the Tiny ImageNet or Open Images datasets
│   │
│   ├── clip_similarity_search.py  # Similar image search module via text, in the Tiny ImageNet or Open Images datasets
│   │
│   ├── frontend/               # Web interface via Streamlit
│   │   ├── main_frontend.py    # Main page           
│   │   ├── research.py         # CBIR research page     
│   │   ├── clip_research.py    # TBIR research page     
│   │   ├── visualization.py    # Visualization page              
│   │   ├── about.py            # About page                     
│
├── ressources/                 # Folder of embeddings, categories, images and links of the two datasets
│   ├── tiny-imagenet/          # Tiny ImageNet folder
│   │   ├── tiny-imagenet-200   # Tiny ImageNet dataset, containing 100k images (to download)
│   │   │
│   │   ├── tinyimagenet_mobilnetv3_feature_extractor.py  # Extractor of vectors from the dataset with MobileNetV3 (for CBIR)
│   │   ├── Tiny_ImageNet_MobilNetV3_Categories.npy       # Image related categories (for CBIR / to download)
│   │   ├── Tiny_ImageNet_MobilNetV3_Embeddings.npy       # Image-related feature vectors (for CBIR / to download)
│   │   │
│   │   ├── tinyimagenet_clip_feature_extractor.py  # Extractor of vectors from the dataset with CLIP (for TBIR)
│   │   ├── Tiny_ImageNet_CLIP_Categories.npy       # Image related categories (for TBIR / to download)
│   │   ├── Tiny_ImageNet_CLIP_Embeddings.npy       # Image-related feature vectors (for TBIR / to download)
│   │
│   ├── open-images/                  # Open Images folder
│   │   ├── images_urls.json          # Table of links pointing to each of the images in Open Images
│   │   │
│   │   ├── clip_embeddings.npy       # Image-related feature vectors (for TBIR / to download)
│   │   ├── mobilenet_embeddings.npy  # Image-related feature vectors (for CBIR / to download)
│
├── test/                       # Unit tests for modules
│   ├── app_test.py             # Test for all modules
│   ├── image_preprocessing_test.py
│   ├── feature_extractor_test.py
│   ├── similarity_search_test.py
│
```

> **IMPORTANT** : Embeddings and category files extracted via MobileNetV3 and CLIP are available on *[this Google Drive](https://drive.google.com/drive/folders/1fG2j6oRhhP7w1kNZm0svfod8yZNfy3pU?usp=share_link)*. These *.npy* files are heavy and cannot be uploaded at the same time as the project.


## Ressources

### Tiny ImageNet
One of the two image sets used is **Tiny ImageNet**. A subset of ImageNet, it is designed for image classification experiments with a small dataset containing 200 classes of 500 images.\
**Source : [CS231N - Stanford](https://cs231n.stanford.edu/) - Téléchargeable depuis ce lien : [tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip).**

### Open Images
The second set of images collected by Google. Compared to Tiny ImageNet, it offers greater visual diversity and significantly higher image quality.\
**Source** : [Google Open Images](https://storage.googleapis.com/openimages/web/index.html)

### MobileNetV3
This project uses **MobileNetV3**, an image classification model developed by Google. This CNN is used for CBIR.\
**Source : [PyTorch MobileNetV3](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)**  

### CLIP - ViT-B/32
This project uses **CLIP**, a multimodal model developed by OpenAI capable of combining images and text for advanced search and classification.\
**Source : [OpenAI CLIP](https://openai.com/index/clip/)**

### FAISS
This project uses **Facebook AI Similarity Search** (FAISS) for indexing and fast search of image vectors.\
**Source : [FAISS.IA](https://faiss.ai/)**


## Unit Tests
Unit tests are launched via the command (for example with *similarity_search_test.py*) and from the project root:
```
python3 -m unittest test/similarity_search_test.py
```

## Use
Before you can test the application, make sure you have downloaded all the necessary resources (dataset, embeddings and categories).\
Next, enter the following command to download the necessary packages:
```
pip install -r requirements.txt
```

The **web application** is launched with the command:
```
streamlit run src/frontend/main_frontend.py
```

To test the application **without interface**:
```
python3 test/app_test.py <mobilenet|clip> <image_path|query_text>
```

## Authors and Other Information

**PixMatcher Team :**
- Nassilya Belguedj
- Aaron Aidoudi
- Julie Colliere
- Mathieu Moustache

**Supervised by**  Mr. Camille Kurtz, as part of the L3E1 project of the third year of the Computer Science degree at Paris Cité University.

Feel free to contribute or raise issues!
