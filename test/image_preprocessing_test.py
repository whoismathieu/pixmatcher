""" Module de test unitaire pour le prétraitement de l'image du fichier image_processing.py. """

import unittest, os, torch
from PIL import Image
from src.image_preprocessing import preprocess_image, is_allowed_extension, InvalidImageFormatError
from pathlib import Path


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """ Configuration initiale avant chaque test. """
        self.valid_image_path = str(Path(__file__).parent.parent / "ressources" / "mountain1.png")
        self.invalid_image_path = "invalid_image.txt"
        self.non_existent_path = "non_existent.jpg"

        # Créer une image de test valide
        image = Image.new('RGB', (500, 500), color='red')
        image.save(self.valid_image_path)

    def tearDown(self):
        """ Nettoyage après chaque test. """
        if os.path.exists(self.valid_image_path):
            os.remove(self.valid_image_path)

    def test_is_allowed_extension_valid(self):
        """ Teste la fonction is_allowed_extension avec une extension valide. """
        self.assertTrue(is_allowed_extension(self.valid_image_path))

    def test_is_allowed_extension_invalid(self):
        """ Teste la fonction is_allowed_extension avec une extension invalide. """
        self.assertFalse(is_allowed_extension(self.invalid_image_path))

    def test_preprocess_image_file_not_found(self):
        """ Vérifie que FileNotFoundError est levé si le fichier n'existe pas. """
        with self.assertRaises(FileNotFoundError):
            preprocess_image(self.non_existent_path)

    def test_preprocess_image_invalid_format(self):
        """ Vérifie que InvalidImageFormatError est levé pour un format non supporté. """
        with open(self.invalid_image_path, 'w') as f:
            f.write("Invalid content")
        with self.assertRaises(InvalidImageFormatError):
            preprocess_image(self.invalid_image_path)
        os.remove(self.invalid_image_path)

    def test_preprocess_image_valid_processing(self):
        """ Teste le prétraitement d'une image valide. """
        processed_image = preprocess_image(self.valid_image_path, target_size=(224, 224), to_tensor=False)
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.size, (224, 224))

    def test_preprocess_image_to_tensor(self):
        """ Teste la conversion en tenseur PyTorch. """
        processed_tensor = preprocess_image(self.valid_image_path, target_size=(224, 224), to_tensor=True)
        self.assertIsInstance(processed_tensor, torch.Tensor)
        self.assertEqual(processed_tensor.shape, (3, 224, 224))  # Vérifie la forme du tenseur

