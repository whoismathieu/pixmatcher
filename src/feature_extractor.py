""" Module d'extraction de caractéristiques basé sur MobileNetV3.

Ce module permet d'extraire un vecteur de caractéristiques (embedding) à partir d'une image,
en s'appuyant sur le modèle MobileNetV3-Large pré-entraîné sur ImageNet. Son fonctionnement est le suivant :

    - L'image est prétraitée soit via un module externe dédié (image_preprocessing.py, si disponible), soit via un
    pipeline de secours basé sur torchvision.
    - Le modèle est tronqué pour ne conserver que les couches convolutionnelles et le pooling global,
    en supprimant la tête de classification, de sorte à obtenir uniquement ce qui est nécessaire.
    - Le résultat est un vecteur 1D de caractéristiques représentant le contenu visuel de l'image.

Ce vecteur est ensuite utilisé pour la recherche de similarité. """

import logging, torch, numpy as np
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

try:  # Tenter d'importer le module de prétraitement personnalisé
    from image_preprocessing import preprocess_image
except ImportError:
    preprocess_image = None
    logging.warning("Module de prétraitement d'image non trouvé. Utilisation d'un prétraitement minimal.")


class FeatureExtractor:
    """ Extrait les caractéristiques d'une image avec MobileNetV3. """

    def __init__(self, device: str = None, target_size: tuple[int, int] = (224, 224)):
        """ Initialise MobileNetV3 en mode évaluation et prépare le pipeline d'extraction des features.
        :param device: 'cuda' ou 'cpu'. Si None, l'appareil est déduit automatiquement.
        :param target_size: Dimension d'entrée exigée par le modèle (largeur, hauteur). """

        if device is None:  # Détermination automatique du device si non spécifié
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1  # Chargement du modèle MobileNetV3 pré-entraîné sur ImageNet
        self.model = mobilenet_v3_large(weights=weights)
        self.model.eval()  # Passage en mode évaluation (désactive dropout, etc.)
        self.model.to(self.device)  # Déplacement du modèle sur le device approprié
        self.target_size = target_size

        # On extrait uniquement les features en supprimant la couche de classification
        self.feature_extractor = torch.nn.Sequential(
            self.model.features,  # Convolutional backbone
            self.model.avgpool,  # Global average pooling
            torch.nn.Flatten()  # Mise à plat du tenseur pour obtenir un vecteur 1D
        )

        # Pipeline de prétraitement minimal en cas d'absence du module personnalisé
        self.basic_transform = Compose([
            Resize(self.target_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],  # Moyennes standard pour les canaux RGB sur ImageNet
                      std=[0.229, 0.224, 0.225])   # Écarts-types standards
        ])

    def extract_features(self, image_input, from_preprocessed: bool = False) -> np.ndarray:
        """ Extrait et retourne le vecteur de caractéristiques de l'image.
        :param image_input: Chemin vers l'image ou objet PIL.Image.
        :param from_preprocessed: Si True, l'image est déjà sous forme de tensor.
        :return: Vecteur 1D des caractéristiques en format numpy. """

        if from_preprocessed:
            image_tensor = image_input  # L'image est supposée déjà prête à l'inférence
        else:
            if preprocess_image:  # Utilisation du module personnalisé de prétraitement
                image_tensor = preprocess_image(image_input, target_size=self.target_size, to_tensor=True)
            else:  # Prétraitement minimal en cas d'absence du module dédié
                try:
                    image = Image.open(image_input).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Erreur lors de l'ouverture de l'image {image_input}: {e}")
                image_tensor = self.basic_transform(image)

        if image_tensor.ndim == 3:  # Si l'image est un tensor 3D (C, H, W), ajoute une dimension batch
            image_tensor = image_tensor.unsqueeze(0)  # Ajoute une dimension en début de tensor pour simuler un batch

        image_tensor = image_tensor.to(self.device)  # Déplace le tensor vers le périphérique spécifié (GPU ou CPU)

        with torch.no_grad():  # Désactive le calcul des gradients pour l'inférence (optimise la mémoire et les calculs)
            features = self.feature_extractor(image_tensor)  # Extraction des caractéristiques via le modèle MobileNetV3

        return features.cpu().numpy().flatten()  # Convertit les caractéristiques en numpy array 1D et les retourne
