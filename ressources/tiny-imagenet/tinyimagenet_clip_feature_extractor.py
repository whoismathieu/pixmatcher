""" Ce programme utilise le modèle CLIP (ViT-B/32) pour extraire les embeddings d'images du dataset Tiny ImageNet.
Il télécharge et extrait le dataset si nécessaire, charge le modèle CLIP, et effectue l'extraction des features en lots
avant de sauvegarder les résultats sous forme de fichiers .npy.
Programme exécuté sur Google Colab. Temps d'exécution : environ 5 heures. """

import torch, clip, numpy as np, os, time, zipfile, requests, shutil
from PIL import Image
from tqdm import tqdm

# Vérifier l'utilisation du GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de {device}")

# URL du dataset Tiny ImageNet et chemin de stockage
TINY_IMAGENET_URL = "cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_PATH = "/content/tiny-imagenet-200"

# Vérifier si le dataset existe déjà, sinon le télécharger et l'extraire
if not os.path.exists(DATASET_PATH):
    print("Téléchargement de Tiny ImageNet...")
    response = requests.get(TINY_IMAGENET_URL, stream=True)
    with open("/content/tiny-imagenet-200.zip", "wb") as file:
        shutil.copyfileobj(response.raw, file)
    print("Extraction des fichiers...")
    with zipfile.ZipFile("/content/tiny-imagenet-200.zip", "r") as zip_ref:
        zip_ref.extractall("/content")
    print("Téléchargement et extraction terminés.")

# Chargement du modèle CLIP ViT-B/32
print("Chargement du modèle CLIP ViT-B/32...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Fonction pour extraire les features d'un batch d'images
def extract_features_batch(batch):
    with torch.no_grad():
        batch = batch.to(device)
        features = model.encode_image(batch)
        features /= features.norm(dim=-1, keepdim=True)  # Normalisation des embeddings
    return features.cpu().numpy()

# Définition des chemins et initialisation des structures de stockage
tiny_imagenet_path = os.path.join(DATASET_PATH, "train")
embeddings = []
categories = []
batch_size = 32
batch_images = []
batch_categories = []

start_time = time.time()

print("Début de l'extraction des embeddings...")
# Parcourir chaque catégorie du dataset
for category in tqdm(os.listdir(tiny_imagenet_path), desc="Traitement des catégories"):
    category_path = os.path.join(tiny_imagenet_path, category, "images")
    if not os.path.isdir(category_path):
        continue

    # Parcourir chaque image de la catégorie
    for image_file in os.listdir(category_path):
        image_path = os.path.join(category_path, image_file)
        try:
            image = preprocess(Image.open(image_path).convert("RGB"))
            batch_images.append(image.unsqueeze(0))
            batch_categories.append(category)

            # Traiter le batch une fois la taille atteinte
            if len(batch_images) == batch_size:
                batch_tensor = torch.cat(batch_images, dim=0).to(device)
                batch_embeddings = extract_features_batch(batch_tensor)
                embeddings.extend(batch_embeddings)
                categories.extend(batch_categories)
                batch_images, batch_categories = [], []
        except Exception as e:
            print(f"Erreur avec {image_path}: {e}")

# Traiter les images restantes dans le dernier batch
if batch_images:
    batch_tensor = torch.cat(batch_images, dim=0).to(device)
    batch_embeddings = extract_features_batch(batch_tensor)
    embeddings.extend(batch_embeddings)
    categories.extend(batch_categories)

# Sauvegarde des embeddings sous format .npy
embeddings = np.array(embeddings)
np.save("/content/Tiny_ImageNet_CLIP_Embeddings.npy", embeddings)
np.save("/content/Tiny_ImageNet_CLIP_Categories.npy", np.array(categories))

end_time = time.time()
print(f"Extraction terminée en {end_time - start_time:.2f} secondes")
print(f"Nombre total d'images traitées : {len(embeddings)}")
print("Embeddings sauvegardés sous 'Tiny_ImageNet_CLIP_Embeddings.npy' et 'Tiny_ImageNet_CLIP_Categories.npy'.")
