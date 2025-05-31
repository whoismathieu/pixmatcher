"""  Module de prétraitement d'image.

Ce module fournit une fonction principale, preprocess_image, utilisée pour préparer des images
avant leur passage dans MobileNetV3, le CNN utilisé. Il prend en charge les étapes suivantes :

  - Vérifie que le fichier existe et que son extension est parmi les formats supportés (JPEG, PNG).
  - Ouvre l'image avec Pillow, et convertit les modes non-RGB en RGB pour assurer la compatibilité.
  - Redimensionne l'image de manière proportionnelle si elle dépasse une taille maximale définie.
  - Redimensionne ensuite l'image à une taille fixe requise par le CNN (par défaut 224x224).
  - Convertit l'image en tenseur PyTorch normalisé (optionnel), avec les statistiques standard d'ImageNet.

Ce module garantit une homogénéité des données d'entrée tout en gérant les cas d'erreurs courants
(fichier introuvable, format non supporté, image illisible). """

import os, logging
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Configuration du système de logs pour afficher des messages informatifs (ex : redimensionnement, conversion)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Seules ces extensions d'image sont acceptées par le système
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']


class InvalidImageFormatError(Exception):
    """ Exception personnalisée pour un format d'image non supporté. """
    pass


def is_allowed_extension(file_path: str) -> bool:
    """ Vérifie si l'extension du fichier image est dans la liste des formats acceptés.
    :param file_path: Chemin du fichier image.
    :return: True si l'extension est acceptable, False sinon. """

    ext = os.path.splitext(file_path)[1].lower()  # Extrait l'extension du fichier et la convertit en minuscules
    return ext in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224), max_dim: int = 1024,
                     to_tensor: bool = True):
    """ Prétraite une image pour la rendre compatible avec l'entrée d'un CNN.
    :param image_path: Chemin vers l'image.
    :param target_size: Taille finale (largeur, hauteur) exigée par le réseau.
    :param max_dim: Taille maximale autorisée pour la plus grande dimension de l'image avant redimensionnement.
    :param to_tensor: Si True, convertit l'image en torch.Tensor et la normalise.
    :return: L'image prétraitée sous forme de PIL.Image ou torch.Tensor selon to_tensor.
    :raises FileNotFoundError: Si le fichier n'existe pas.
    :raises InvalidImageFormatError: Si le format de l'image n'est pas supporté.
    :raises ValueError: Si l'image ne peut pas être ouverte. """

    if not os.path.exists(image_path):  # Vérification de l'existence du fichier
        raise FileNotFoundError(f"L'image spécifiée n'existe pas : {image_path}")

    if not is_allowed_extension(image_path):  # Vérification de l'extension du fichier
        raise InvalidImageFormatError(f"Format de fichier non supporté : {image_path}")

    try:
        image = Image.open(image_path)  # Ouverture de l'image
    except UnidentifiedImageError as e:
        raise ValueError(f"Impossible d'ouvrir l'image {image_path}. Erreur : {e}")

    if image.mode != 'RGB':  # Conversion en RGB si l'image n'est pas déjà dans ce mode
        logging.info(f"Conversion de l'image en RGB (mode initial : {image.mode}).")
        image = image.convert('RGB')

    width, height = image.size  # Récupération des dimensions actuelles de l'image
    # Si une des dimensions dépasse la limite autorisée (max_dim), on redimensionne
    # l'image en conservant le ratio (aspect ratio) pour ne pas la déformer.
    if max(width, height) > max_dim:
        scaling_factor = max_dim / max(width, height)  # Rapport de réduction à appliquer
        new_size = (int(width * scaling_factor), int(height * scaling_factor))  # Nouvelles dimensions calculées
        logging.info(f"Image trop grande ({width}x{height}). Redimensionnement à {new_size} pour conserver l'aspect.")
        image = image.resize(new_size, Image.LANCZOS)  # Redimensionnement avec un filtre haute qualité (LANCZOS)

    # Redimensionnement strict à la taille cible requise par le CNN
    # Cela permet de standardiser toutes les entrées, quel que soit le format d'origine.
    if image.size != target_size:
        logging.info(f"Redimensionnement final de l'image à la taille cible : {target_size}.")
        image = image.resize(target_size, Image.LANCZOS)

    # Si demandé, conversion en tenseur PyTorch suivi d'une normalisation avec les moyennes/écarts-types
    # utilisés lors de l'entraînement des modèles ImageNet.
    # Cette étape est indispensable pour que les entrées soient compatibles avec les modèles pré-entraînés.
    if to_tensor:
        to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = to_tensor_transform(image)

    return image
