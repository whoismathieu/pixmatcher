""" Module de recherche d'images similaires et de catégorisation avec FAISS.

Ce module implémente des fonctions permettant de :

    - Charger les embeddings et les catégories des datasets Tiny ImageNet et Open Images.
    - Construire un index FAISS optimisé pour une recherche rapide de similarité basée sur des embeddings d'images.
    - Trouver les 5 catégories les plus proches d'une image donnée, en utilisant la distance cosine pour la comparaison
    des embeddings.
    - Trouver les k images les plus semblables à une image donnée, grâce à l'index FAISS.
    - Retrouver le chemin d'une image à partir de son index dans les datasets, facilitant l'accès direct aux images
    correspondantes.

Il permet une recherche d'images rapides et efficaces en s'appuyant sur la structure d'indexation FAISS, tout en prenant
en charge deux datasets : Open Images et Tiny ImageNet. """

import numpy as np, os, faiss, json
from pathlib import Path
from scipy.spatial.distance import cosine

""" --------------------- Partie Open Image V7 ---------------------- """

# Charger le mapping index -> nom d'image depuis le fichier JSON
with open(Path(__file__).parent.parent / "ressources" / "open-images" / "image_urls.json", "r") as f:
    image_urls = json.load(f)

# Définir l'URL de base pour accéder aux images sur AWS
base_url = "https://pixmatcher-images.s3.eu-west-3.amazonaws.com/"

# Charger les embeddings CLIP pour Open Image V7 depuis un fichier .npy
OI_EMBEDDINGS_PATH = (np.load(Path(__file__).parent.parent / "ressources" / "open-images" / "mobilenet_embeddings.npy")
                      .astype('float32'))

# Construire un index FAISS pour Open Image V7 basé sur les embeddings
dimension = OI_EMBEDDINGS_PATH.shape[1]  # Extraire la dimension des vecteurs d'embedding
oi_index = faiss.IndexFlatL2(dimension)  # Créer un index basé sur la distance L2 (euclidienne)
oi_index.add(OI_EMBEDDINGS_PATH)  # Ajouter les embeddings à l'index FAISS


def oi_find_top_similar_images(image_features: np.ndarray, k):
    """ Trouve les k images les plus similaires dans Open Images à partir d'un vecteur de caractéristiques avec FAISS.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des indices des k images les plus similaires et leurs distances """

    # Convertit les caractéristiques de l'image en type 'float32' et les redimensionne en un vecteur 1D si nécessaire.
    # Permet d'assurer que les données sont compatibles avec FAISS qui attend des vecteurs 1D pour chaque image.
    image_features = image_features.astype('float32').reshape(1, -1)

    # Vérification de la dimension du vecteur d'image par rapport à l'index FAISS
    # FAISS utilise un index qui a été construit avec une dimension spécifique pour les embeddings.
    if image_features.shape[1] != oi_index.d:
        raise ValueError(
            f"Erreur : la dimension de l'image ({image_features.shape[1]}) ne correspond pas à la dimension FAISS "
            f"({oi_index.d})")

    # Lancer la recherche FAISS pour les k images les plus similaires
    # Utilise l'index FAISS pour rechercher les k images les plus proches de l'image donnée,
    # en fonction de la distance L2 (Euclidienne). FAISS renvoie les distances et les indices des images semblables.
    # 'distances' contient les distances entre l'image donnée et chaque image trouvée,
    # et 'indices' contient les indices des images dans l'index FAISS.
    distances, indices = oi_index.search(image_features, k)
    # La liste contient des tuples (index de l'image, distance de similarité) pour les k images les plus proches.
    top_k_similar = [(idx, distances[0][i]) for i, idx in enumerate(indices[0])]

    return top_k_similar


def oi_get_image_path(index_image):
    """ Retrouve l'URL complète de l'image (d'Open Images) à partir de son index.
    :param index_image: Index de l'image dans les embeddings
    :return: URL complète de l'image """

    # Récupérer le nom du fichier correspondant à l'index dans le mapping
    filename = image_urls.get(str(index_image))

    if filename:
        return base_url + filename
    else:
        return None


""" --------------------- Partie Tiny ImageNet ---------------------- """

# Charger les embeddings et catégories Tiny ImageNet
TINY_IMAGENET_PATH = Path(__file__).parent.parent / "ressources" / "tiny-imagenet" / "tiny-imagenet-200"
TI_EMBEDDINGS_PATH = np.load(Path(__file__).parent.parent / "ressources" / "tiny-imagenet"
                             / "Tiny_ImageNet_MobilNetV3_Embeddings.npy").astype('float32')
TI_CATEGORIES_PATH = np.load(Path(__file__).parent.parent / "ressources" / "tiny-imagenet"
                             / "Tiny_ImageNet_MobilNetV3_Categories.npy", allow_pickle=True)

# Construire l'index FAISS pour Tiny ImageNet basé sur les embeddings
dimension = TI_EMBEDDINGS_PATH.shape[1]  # Taille des vecteurs d'embeddings
ti_index = faiss.IndexFlatL2(dimension)  # Créer un index basé sur la distance L2
ti_index.add(TI_EMBEDDINGS_PATH)  # Ajouter les embeddings à l'index FAISS


def ti_find_top5_categories(image_features: np.ndarray):
    """ Trouve les 5 catégories les plus similaires à partir d'un vecteur de caractéristiques.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des 5 catégories les plus similaires """

    # Cette section charge le fichier "words.txt" qui contient la correspondance entre les identifiants numériques
    # des classes dans Tiny ImageNet et leurs labels respectifs. Chaque ligne du fichier contient un identifiant de
    # classe et son label séparés par une tabulation.
    wordnet_mapping = {}
    with open(TINY_IMAGENET_PATH / "words.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wordnet_mapping[parts[0]] = parts[1]

    # Ajuster la taille du vecteur de caractéristiques à la taille des embeddings
    # Si la dimension du vecteur d'image est trop grande, elle est tronquée à la taille des embeddings
    if image_features.shape[0] > TI_EMBEDDINGS_PATH.shape[1]:
        image_features = image_features[:TI_EMBEDDINGS_PATH.shape[1]]
    # Si elle est trop petite, elle est complétée par des zéros (padding)
    elif image_features.shape[0] < TI_EMBEDDINGS_PATH.shape[1]:
        image_features = np.pad(image_features, (0, TI_EMBEDDINGS_PATH.shape[1] - image_features.shape[0]))

    # La distance cosine est utilisée ici pour mesurer la similarité entre les caractéristiques de l'image et les
    # embeddings Tiny ImageNet. Une distance plus faible indique une plus grande similarité entre l'image et l'embedding
    distances = [cosine(image_features, emb) for emb in TI_EMBEDDINGS_PATH]
    top_5_indices = np.argsort(distances)[:5]  # Récupérer les indices des 5 catégories les plus proches

    # Extraire les labels des catégories correspondantes
    top_5_categories = [(wordnet_mapping.get(TI_CATEGORIES_PATH[idx], "Inconnu"), distances[idx])
                        for idx in top_5_indices]

    return top_5_categories


def ti_find_top_similar_images(image_features: np.ndarray, k):
    """ Trouve les k images les plus similaires à partir d'un vecteur de caractéristiques avec FAISS.
    :param image_features: Vecteur de caractéristiques de l'image
    :return: Liste des indices des k images les plus similaires et leurs distances """

    # Vérifier la dimension de l'image avant la recherche FAISS
    print(f"Dimension réelle de l'image en entrée: {image_features.shape}")
    print(f"Dimension attendue par FAISS: {ti_index.d}")

    # Assurer que l'image est bien un vecteur 1D
    image_features = image_features.astype('float32').reshape(1, -1)

    # Vérifier la taille avant la recherche
    if image_features.shape[1] != ti_index.d:
        print(f"Erreur : la dimension de l'image ({image_features.shape[1]}) ne correspond pas à la dimension FAISS "
              f"({ti_index.d})")

    # Lancer la recherche FAISS
    # Exécution de la recherche des k voisins les plus proches dans l'espace vectoriel des embeddings
    # à l'aide de l'index FAISS basé sur la distance euclidienne (L2).
    distances, indices = ti_index.search(image_features, k)

    # Créer une liste des k images les plus similaires avec leur distance
    # On associe chaque indice retourné par FAISS avec sa distance correspondante,
    # ce qui permet d'identifier les images les plus proches ainsi que leur niveau de similarité.
    top_k_similar = [(idx, distances[0][i]) for i, idx in enumerate(indices[0])]

    return top_k_similar


def ti_get_image_path(index_image, base_path=TINY_IMAGENET_PATH):
    """ Retrouve le chemin de l'image à partir de son index.
    :param index_image: Index de l'image dans le fichier d'embeddings
    :param base_path: Chemin de base vers le dataset Tiny ImageNet
    :return: Chemin complet vers l'image """

    # Identifier la classe et l'indice de l'image dans le dataset Tiny ImageNet
    class_id = TI_CATEGORIES_PATH[index_image]
    class_indices = np.where(TI_CATEGORIES_PATH == class_id)[0]
    image_index = np.where(class_indices == index_image)[0][0]

    # Construire le nom de l'image et son chemin dans le dataset
    image_name = f"{class_id}_{image_index}.JPEG"
    image_path = os.path.join(base_path, "train", class_id, "images", image_name)

    return image_path
