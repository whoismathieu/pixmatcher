""" Module de test unitaire pour la recherche de siilarité du fichier similarity_search.py. """

import unittest, numpy as np
from unittest.mock import patch
from src.similarity_search import ti_find_top5_categories, ti_get_image_path
from pathlib import Path


class TestSimilaritySearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Configuration initiale pour les tests. """
        # Simuler des embeddings Tiny ImageNet
        cls.simulated_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.1, 0.3, 0.5],
            [0.6, 0.4, 0.2]
        ], dtype='float32')

        # Simuler des catégories Tiny ImageNet
        cls.simulated_categories = np.array(['n01443537', 'n01629276', 'n01641577', 'n01644900', 'n01698640'])

        # Simuler le mapping WordNet
        cls.simulated_wordnet_mapping = {
            'n01443537': 'goldfish, Carassius auratus',
            'n01629276': 'salamander',
            'n01641577': 'bullfrog, Rana catesbeiana',
            'n01644900': 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
            'n01698640': 'American alligator, Alligator mississipiensis'
        }

    @patch("src.similarity_search.EMBEDDINGS_PATH", new_callable=lambda: TestSimilaritySearch.simulated_embeddings)
    @patch("src.similarity_search.CATEGORIES_PATH", new_callable=lambda: TestSimilaritySearch.simulated_categories)

    def test_find_top5_categories(self, mock_embeddings, mock_categories):
        """ Teste la fonction find_top5_categories avec un vecteur de caractéristiques simulé. """
        # Simuler un vecteur de caractéristiques
        query_features = np.array([0.15, 0.25, 0.35], dtype='float32')

        # Appeler la fonction à tester
        top5_categories = ti_find_top5_categories(query_features)

        # Vérifier que la sortie contient exactement 5 éléments
        self.assertEqual(len(top5_categories), 5)

        # Vérifier que chaque élément contient une catégorie et une distance
        for category, distance in top5_categories:
            self.assertIsInstance(category, str)
            self.assertIsInstance(distance, float)

        # Vérifier que les catégories retournées sont dans le mapping simulé
        for category, _ in top5_categories:
            self.assertIn(category, self.simulated_wordnet_mapping.values())

    @patch("src.similarity_search.CATEGORIES_PATH", new_callable=lambda: TestSimilaritySearch.simulated_categories)

    def test_get_image_path(self,mock_categories):
        """ Teste la fonction get_image_path avec des indices simulés et un chemin de base simulé. """

        test_index = 2  # Index arbitraire pour le test
        image_path = ti_get_image_path(test_index)  # Appeler la fonction à tester
        base_path = Path(__file__).parent.parent / "ressources" / "tiny-imagenet-200"
        # Construire le chemin attendu
        expected_class_id = self.simulated_categories[test_index]
        expected_image_name = f"{expected_class_id}_0.JPEG"
        expected_path = f"{base_path}/train/{expected_class_id}/images/{expected_image_name}"

        # Vérifier que le chemin retourné correspond au chemin attendu
        self.assertEqual(image_path, expected_path)
