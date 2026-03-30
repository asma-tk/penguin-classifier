import os
import time
import requests
from tqdm import tqdm #barre de progression visuelle
from PIL import Image
from io import BytesIO
from config import *

# PARAMÈTRES
MAX_IMAGES = 250
PAUSE = 1        #       attend 1 seconde entre chaque page
MAX_RETRIES = 5  # Nombre de tentatives en cas d'erreur réseau
API_URL = "https://api.inaturalist.org/v1/observations" ## URL de base de l’API

# API iNaturalist avec gestion des erreurs réseau
def fetch_page(taxon, page_number, limit=30):    # fonction pour récupérer une page d’observations
    payload = {                                 # paramètres envoyés à l’API
        "taxon_name": taxon,                    # nom scientifique de l’espèce
        "quality_grade": "research",            # seules les observations fiables
        "photos": True,                          # seulement les observations avec photo
        "page": page_number,                     # numéro de page à récupérer
        "per_page": limit,                       # nombre d’éléments par page
        "order_by": "created_at",               # tri par date de création
        "order": "desc"                          # ordre décroissant (les plus récentes d’abord)
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(API_URL, params=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("results", [])
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt < MAX_RETRIES - 1: #Si ce n’est pas la dernière tentative, on va réessayer.
                wait = (attempt + 1) * 5 #attend 5 s, deuxième échec → 10 s, troisième → 15 s, etc.
                print(f"\n⚠️ Erreur réseau : {e}. Nouvel essai dans {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n❌ Échec définitif pour la page {page_number}")
                return []

def save_image(image_url, destination):
    # On ne télécharge pas si le fichier existe déjà
    if os.path.exists(destination):
        return True
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(destination, format="JPEG", quality=95)
    return True

# TÉLÉCHARGEMENT PAR ESPÈCE
def collect_species(species_name, latin_name):
    print(f"\n🐧 Espèce en cours : {species_name} ({latin_name})")
    target_dir = os.path.join(DATASET_DIR, species_name)  
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    current_page = 1
    
    # Compter combien d'images on a déjà pour reprendre là où on s'est arrêté
    existing_files = len([f for f in os.listdir(target_dir) if f.endswith('.jpg')])
    count = existing_files
    
    with tqdm(total=MAX_IMAGES, initial=count) as bar:   #  affichage dans le prompt
        while count < MAX_IMAGES:
            data = fetch_page(latin_name, current_page)
            if not data:
                print(f"⚠️ Plus de données ou erreur pour {species_name}")
                break
            
            for observation in data:
                if count >= MAX_IMAGES:    #.   Arrête si on a 250 images
                    break
                
                photos = observation.get("photos")
                if not photos:
                    continue
                
                original_url = photos[0]["url"].replace("square", "original")  #pour avoir la haute résolution.
                filename = f"{species_name}_{count:05d}.jpg"
                filepath = os.path.join(target_dir, filename)
                
                try:
                    if save_image(original_url, filepath):
                        count += 1
                        bar.update(1)
                except Exception:
                    continue  # Ignore les images corrompues ou URLs mortes
            
            current_page += 1
            time.sleep(PAUSE)
    
    print(f" ✅ {count} images prêtes pour {species_name}")

# PROGRAMME PRINCIPAL
if __name__ == "__main__":                   #s'assure que cette partie s'exécute seulement si le fichier est lancé directement
    os.makedirs(DATASET_DIR, exist_ok=True)  #  Utilise DATASET_DIR   
    for common_name, scientific_name in PENGUINS.items():
        collect_species(common_name, scientific_name)
    
    print("\n🎉 Dataset prêt pour l'entraînement !")