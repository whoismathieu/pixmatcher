""" Programme permettant d'extraire les vecteurs de caractéristiques du jeu de données Tiny ImageNet ave MobileNetV3. Le processus inclut les étapes suivantes :
- Téléchargement du jeu de données Tiny ImageNet puis prétraitement des images.
- Chargement de MobileNetV3 puis suppression de la dernière couche afin d'obtenir des embeddings intermédiaires.
- Extraction des embeddings par lots pour optimiser les performances.
- Sauvegarde des embeddings et des catégories correspondantes dans des fichiers numpy.
Programme exécuté sur Google Colab. Temps d'exécution : environ 1 heure. """


import torch, numpy as np, os, time, zipfile, requests, shutil
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Vérification de la disponibilité d'un GPU et définition du périphérique à utiliser (GPU ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de {device}")

# Téléchargement et extraction automatique du jeu de données Tiny ImageNet
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_PATH = "/content/tiny-imagenet-200"

if not os.path.exists(DATASET_PATH):
    print("Téléchargement de Tiny ImageNet...")
    response = requests.get(TINY_IMAGENET_URL, stream=True)
    with open("/content/tiny-imagenet-200.zip", "wb") as file:
        shutil.copyfileobj(response.raw, file)

    print("Extraction des fichiers...")
    with zipfile.ZipFile("/content/tiny-imagenet-200.zip", "r") as zip_ref:
        zip_ref.extractall("/content")

    print("Téléchargement et extraction terminés.")

# Chargement du modèle MobileNetV3, avec les poids large
print("Chargement du modèle MobileNetV3...")
model = models.mobilenet_v3_large(weights='IMAGENET1K_V1').to(device)
model.eval()

# Suppression de la dernière couche pour obtenir les embeddings intermédiaires (vecteurs de caractéristiques)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

# Définition des transformations à appliquer aux images avant leur traitement par le modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionnement à 224x224 pixels
    transforms.ToTensor(),          # Conversion en tenseur PyTorch
    transforms.Normalize(           # Normalisation selon les moyennes et écarts-types d'ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Fonction pour extraire les embeddings d'un lot d'images en utilisant le modèle pré-entraîné
def extract_features_batch(batch):
    with torch.no_grad():  # Désactivation du calcul de gradient pour accélérer le calcul
        batch = batch.to(device)  # Envoi du lot d'images vers le GPU ou CPU utilisé
        features = feature_extractor(batch)  # Extraction des caractéristiques intermédiaires
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Pooling adaptatif moyen global
        features = features.view(features.size(0), -1)  # Aplatir les caractéristiques en vecteurs unidimensionnels
    return features.cpu().numpy()  # Retourner les vecteurs sous forme numpy sur CPU

# Extraction effective des embeddings pour toutes les images du jeu de données Tiny ImageNet (dossier "train")
tiny_imagenet_path = os.path.join(DATASET_PATH, "train")
embeddings = []      # Liste pour stocker les embeddings extraits
categories = []      # Liste pour stocker les catégories correspondantes aux embeddings
batch_size = 32      # Taille du lot pour optimiser l'extraction par batchs (lots)
batch_images = []    # Liste temporaire pour accumuler les images d'un lot avant traitement
batch_categories = []  # Liste temporaire pour accumuler les catégories correspondantes

start_time = time.time()  # Début du chronomètre pour mesurer la durée totale d'extraction

print("Début de l'extraction des embeddings...")
for category in tqdm(os.listdir(tiny_imagenet_path), desc="Traitement des catégories"):
    category_path = os.path.join(tiny_imagenet_path, category, "images")
    if not os.path.isdir(category_path):
        continue  # Ignorer si le chemin n'est pas un dossier valide

    for image_file in os.listdir(category_path):
        image_path = os.path.join(category_path, image_file)
        try:
            image = Image.open(image_path).convert("RGB")       # Ouverture et conversion en RGB de l'image
            image_tensor = transform(image).unsqueeze(0)        # Application des transformations définies précédemment et ajout d'une dimension batch
            batch_images.append(image_tensor)                   # Ajout à la liste temporaire d'images du lot actuel
            batch_categories.append(category)                   # Ajout à la liste temporaire des catégories correspondantes

            if len(batch_images) == batch_size:                 # Lorsque le lot atteint la taille définie :
                batch_tensor = torch.cat(batch_images, dim=0)   # Concaténation en un seul tenseur PyTorch (batch complet)
                batch_embeddings = extract_features_batch(batch_tensor)  # Extraction effective des embeddings pour ce lot
                embeddings.extend(batch_embeddings)             # Stockage permanent des embeddings extraits dans la liste principale
                categories.extend(batch_categories)             # Stockage permanent des catégories correspondantes dans la liste principale
                batch_images, batch_categories = [], []         # Réinitialisation des listes temporaires pour le prochain lot

        except Exception as e:
            print(f"Erreur avec {image_path}: {e}")             # Affichage d'une erreur éventuelle sans interrompre le processus global

# Traitement éventuel du dernier lot incomplet restant après la boucle principale (si nécessaire)
if batch_images:
    batch_tensor = torch.cat(batch_images, dim=0)
    batch_embeddings = extract_features_batch(batch_tensor)
    embeddings.extend(batch_embeddings)
    categories.extend(batch_categories)

# Sauvegarde finale des embeddings extraits et catégories associées sous forme de fichiers numpy (.npy)
embeddings = np.array(embeddings)
np.save("/content/tiny_imagenet_embeddings.npy", embeddings)
np.save("/content/tiny_imagenet_categories.npy", np.array(categories))

end_time = time.time()  # Fin du chronomètre

# Affichage récapitulatif final : temps total écoulé et nombre total d'images traitées
print(f"Extraction terminée en {end_time - start_time:.2f} secondes")
print(f"Nombre total d'images traitées : {len(embeddings)}")
print("Embeddings sauvegardés sous 'tiny_imagenet_embeddings.npy' et 'tiny_imagenet_categories.npy'.")

# Téléchargement optionnel direct depuis Google Colab vers votre ordinateur local
from google.colab import files
files.download("/content/tiny_imagenet_embeddings.npy")
files.download("/content/tiny_imagenet_categories.npy")
