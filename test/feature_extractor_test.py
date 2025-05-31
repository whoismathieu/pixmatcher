""" Module de test unitaire pour l'extraction du vecteur de caractéristique du fichier feature_extractor.py. """

import unittest, os, torch, numpy as np
from PIL import Image
from src.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """ Configuration initiale avant chaque test. """
        self.device = "cpu"  # Utilisation du CPU pour les tests
        self.target_size = (224, 224)
        self.extractor = FeatureExtractor(device=self.device, target_size=self.target_size)

        # Création d'une image de test valide
        self.valid_image_path = "test_image.jpg"
        image = Image.new('RGB', (500, 500), color='blue')
        image.save(self.valid_image_path)

    def tearDown(self):
        """ Nettoyage après chaque test. """
        if os.path.exists(self.valid_image_path):
            os.remove(self.valid_image_path)

    def test_initialization(self):
        """ Teste l'initialisation du FeatureExtractor. """
        self.assertEqual(self.extractor.device, self.device)
        self.assertEqual(self.extractor.target_size, self.target_size)
        self.assertTrue(isinstance(self.extractor.feature_extractor, torch.nn.Sequential))

    def test_extract_features_from_valid_image(self):
        """ Teste l'extraction des caractéristiques à partir d'une image valide. """
        features = self.extractor.extract_features(self.valid_image_path)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)  # Le vecteur doit être 1D

    def test_extract_features_from_preprocessed_tensor(self):
        """ Teste l'extraction des caractéristiques à partir d'un tenseur prétraité. """
        # Création d'un tenseur prétraité simulé
        image_tensor = torch.rand(1, 3, *self.target_size)  # Batch size = 1
        features = self.extractor.extract_features(image_tensor, from_preprocessed=True)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)

    def test_invalid_image_path(self):
        """ Vérifie qu'une erreur est levée pour un chemin d'image invalide. """
        with self.assertRaises(ValueError):
            self.extractor.extract_features("invalid_path.jpg")

    def test_invalid_image_format(self):
        """ Vérifie qu'une erreur est levée pour un format d'image non supporté. """
        invalid_image_path = "invalid_image.txt"

        with open(invalid_image_path, 'w') as f:
            f.write("Invalid content")

        with self.assertRaises(ValueError):
            self.extractor.extract_features(invalid_image_path)

        os.remove(invalid_image_path)

    def test_device_transfer(self):
        """ Vérifie que le modèle est correctement transféré sur le bon appareil. """
        if torch.cuda.is_available():
            extractor_cuda = FeatureExtractor(device="cuda")
            self.assertEqual(extractor_cuda.device, "cuda")
            extractor_cuda.model.to("cpu")  # Nettoyage manuel si nécessaire

