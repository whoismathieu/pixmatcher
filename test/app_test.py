""" Module de test général. Usage : python test_script.py <mobilenet|clip> <image_path|query_text>

Ce script permet de tester deux modèles de recherche d'images similaires :

    1. MobileNet, qui extrait des caractéristiques à partir d'une image donnée et trouve les images les plus proches.
    2. CLIP, qui permet de rechercher des images à partir d'une requête textuelle en utilisant des embeddings
    pré-calculés. """

import logging
import sys
from PIL import Image
from matplotlib import pyplot as plt
from src.image_preprocessing import preprocess_image
from src.feature_extractor import FeatureExtractor
from src.similarity_search import oi_find_top_similar_images, oi_get_image_path
from src.clip_similarity_search import oi_find_similar_images


def display_images(indices):
    """ Affiche les images correspondant aux indices donnés sur deux rangées horizontales.
    :param indices: Liste des indices des images à afficher. """

    num_images = len(indices)
    num_rows = 2  # Nombre de rangées
    num_cols = (num_images + 1) // 2  # Calcul du nombre de colonnes (arrondi vers le haut)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))  # Création de la figure avec plusieurs images
    axes = axes.flatten()  # Aplatit les axes pour un accès facile, même si certaines cases restent vides

    for i, idx in enumerate(indices):
        img_path = oi_get_image_path(idx)  # Récupère le chemin de l'image
        image = Image.open(img_path)  # Ouvre l'image
        axes[i].imshow(image)  # Affiche l'image
        axes[i].axis("off")  # Supprime les axes pour une meilleure lisibilité

    # Désactiver les cases inutilisées si le nombre d'images est impair
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()  # Ajuste automatiquement les espacements entre les sous-graphiques
    plt.show()


def main(image_path=None, query_text=None, model_type="mobilenet"):
    """ Fonction principale qui gère l'exécution du programme en fonction du modèle choisi.
        Elle effectue le prétraitement, l'extraction de caractéristiques et la recherche des images similaires.
        :param image_path & query_text: Chemin de l'image (pour MobileNet) ou texte de la requête (pour CLIP).
        :param model_type: Type de modèle à utiliser ("mobilenet" ou "clip"). """

    try:
        if model_type == "mobilenet":
            if image_path is None:
                raise ValueError("Un chemin d'image est requis pour MobileNet.")

            # Prétraitement de l’image (mise à l’échelle, tensorisation…)
            processed_image = preprocess_image(image_path, target_size=(224, 224), to_tensor=True)

            # Extraction des caractéristiques avec MobileNet
            extractor = FeatureExtractor()
            features = extractor.extract_features(processed_image, from_preprocessed=True)
            print("Vecteur de caractéristiques:", features.shape)

            # Recherche des images les plus similaires
            top_images = oi_find_top_similar_images(features, 10)

        elif model_type == "clip":
            if query_text is None:
                raise ValueError("Un texte de requête est requis pour CLIP.")

            # Recherche textuelle avec CLIP (via similarité d’embedding)
            top_images = oi_find_similar_images(query_text)

        else:
            raise ValueError("Modèle inconnu. Choisissez 'mobilenet' ou 'clip'.")

        print("Top images les plus similaires :", top_images)
        display_images(top_images)

    except Exception as e:
        logging.error(f"Erreur : {e}")


if __name__ == "__main__":
    # Vérifie les arguments passés en ligne de commande
    if len(sys.argv) > 2:
        mode = sys.argv[1].lower()
        if mode == "mobilenet":
            image_path = sys.argv[2]
            main(image_path=image_path, model_type="mobilenet")
        elif mode == "clip":
            query_text = sys.argv[2]
            main(query_text=query_text, model_type="clip")
        else:
            print("Usage : python test_script.py <mobilenet|clip> <image_path|query_text>")
    else:
        # Mode par défaut si aucun argument n'est fourni
        print("Usage : python test_script.py <mobilenet|clip> <image_path|query_text>")
        print("Exemple avec la query 'this is a cat' :")
        main(query_text='this is a cat', model_type="clip")
