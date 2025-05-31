""" Module de recherche d’images par similarité textuelle avec CLIP.

Ce programme utilise le modèle CLIP (Contrastive Language-Image Pretraining) pour effectuer une recherche
d’images à partir d’une requête en langage naturel. Il permet de :

    - Charger les embeddings CLIP des images des datasets Tiny ImageNet et Open Images V7
    - Transformer une requête textuelle en vecteur d’embedding via CLIP
    - Comparer ce vecteur aux vecteurs d’images pour identifier les plus similaires
    - Obtenir les indices ou chemins (locaux ou URLs) des images correspondantes

Le modèle utilisé est CLIP ViT-B/32, pré-entraîné et exploité ici en inférence (sans apprentissage). """

import torch, clip, numpy as np, json, os
from scipy.spatial.distance import cdist
from pathlib import Path

# Sélection du device pour l'inférence (CUDA si disponible, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du modèle CLIP (version ViT-B/32) et du préprocesseur associé
model, preprocess = clip.load("ViT-B/32", device=device)


def text_to_vector(text):
    """ Encode une requête textuelle en vecteur d'embedding avec CLIP.
    :param text: Chaîne de texte à encoder
    :return: Vecteur numpy normalisé représentant la requête textuelle """

    with torch.no_grad(): # Pas de gradients car inférence
        # Tokenisation puis encodage du texte avec CLIP
        text_features = model.encode_text(clip.tokenize([text]).to(device))
        # Normalisation du vecteur
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


""" --------------------- Partie Open Image V7 ---------------------- """

# Chemin vers les embeddings CLIP des images Open Image V7
OI_EMBEDDINGS_PATH = np.load(Path(__file__).parent.parent / "ressources" / "open-images" / "clip_embeddings.npy")

# Chargement du mapping entre index FAISS et nom de fichier image
with open(Path(__file__).parent.parent / "ressources" / "open-images" / "image_urls.json", "r") as f:
    image_urls = json.load(f)

# URL de base pour accéder aux images stockées sur S3
BASE_URL = "https://pixmatcher-images.s3.eu-west-3.amazonaws.com/"

def oi_find_similar_images(text, top_k=10):
    """ Trouve les top_k images les plus similaires à partir d'une requête textuelle dans Open Image V7.
    :param text: Texte de la requête utilisateur
    :param top_k: Nombre d’images similaires à retourner
    :return: Liste des indices des images les plus proches """

    query_vector = text_to_vector(text)  # Encodage de la requête

    # Calcul du score de similarité par produit scalaire (vecteurs déjà normalisés)
    similarities = (query_vector @ OI_EMBEDDINGS_PATH.T)[0]

    # Retourne les indices des images les plus proches (tri décroissant)
    return np.argsort(-similarities)[:top_k]


def oi_get_image_path(index):
    """ Retrouve l’URL complète d’une image Open Image V7 à partir de son index.
    :param index: Index de l’image dans les embeddings
    :return: URL S3 de l’image """

    filename = image_urls.get(str(index))
    if filename:
        return BASE_URL + filename
    else:
        return None


""" --------------------- Partie Tiny ImageNet ---------------------- """

# Chargement des embeddings et catégories CLIP des images Tiny ImageNet
TI_EMBEDDINGS_PATH = np.load(Path(__file__).parent.parent / "ressources" / "tiny-imagenet"
                             / "Tiny_ImageNet_CLIP_Embeddings.npy")
CATEGORIES_PATH = np.load(Path(__file__).parent.parent / "ressources" / "tiny-imagenet"
                          / "Tiny_ImageNet_CLIP_Categories.npy", allow_pickle=True)

# Chemin de base vers les images locales Tiny ImageNet
BASE_PATH = Path(__file__).parent.parent / "ressources" / "tiny-imagenet" / "tiny-imagenet-200" / "train"


def ti_find_similar_images(text, top_k=10):
    """ Trouve les top_k images les plus similaires à partir d'une requête textuelle.
    :param text: Texte de la requête.
    :param top_k: Nombre d'images à retourner (par défaut 5).
    :return: Indices des images les plus similaires. """

    query_vector = text_to_vector(text)  # Encodage du texte en vecteur

    # Calcul des distances cosinus entre la requête et chaque embedding image
    distances = cdist(query_vector, TI_EMBEDDINGS_PATH, metric="cosine")[0]

    return np.argsort(distances)[:top_k]  # Indices des images les plus proches


def ti_get_image_path(index):
    """ Retrouve le chemin de l'image à partir de son index.
    :param index: Index de l'image dans le fichier d'embeddings
    :return: Chemin complet vers l'image """

    class_id = CATEGORIES_PATH[index]  # ID WordNet de la classe de l’image
    class_indices = np.where(CATEGORIES_PATH == class_id)[0]  # Tous les indices de la même classe
    image_index = np.where(class_indices == index)[0][0]  # Trouve l’indice dans la sous-liste

    image_name = f"{class_id}_{image_index}.JPEG"
    image_path = os.path.join(BASE_PATH, class_id, "images", image_name)

    return image_path
