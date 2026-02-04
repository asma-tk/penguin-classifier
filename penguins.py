import os
import time
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from config import PENGUINS

# PARAMÈTRES

BASE_DIR = "dataset_original/train/penguins"
MAX_IMAGES = 50
PAUSE = 1

API_URL = "https://api.inaturalist.org/v1/observations"


# API iNaturalist

def fetch_page(taxon, page_number, limit=30):
    payload = {
        "taxon_name": taxon,
        "quality_grade": "research",
        "photos": True,
        "page": page_number,
        "per_page": limit,
        "order_by": "created_at",
        "order": "desc"
    }

    response = requests.get(API_URL, params=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def save_image(image_url, destination):
    response = requests.get(image_url, timeout=30)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(destination, format="JPEG", quality=95)


# TÉLÉCHARGEMENT PAR ESPÈCE

def collect_species(species_name, latin_name):
    print(f"\n🐧 Espèce en cours : {species_name} ({latin_name})")

    target_dir = os.path.join(BASE_DIR, species_name)
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    current_page = 1

    with tqdm(total=MAX_IMAGES) as bar:
        while count < MAX_IMAGES:
            data = fetch_page(latin_name, current_page)

            if len(data) == 0:
                print("⚠️ Plus de données disponibles")
                break

            for observation in data:
                if count >= MAX_IMAGES:
                    break

                photos = observation.get("photos")
                if not photos:
                    continue

                original_url = photos[0]["url"].replace("square", "original")
                filename = f"{species_name}_{count:05d}.jpg"
                filepath = os.path.join(target_dir, filename)

                try:
                    save_image(original_url, filepath)
                    count += 1
                    bar.update(1)
                except Exception:
                    pass

            current_page += 1
            time.sleep(PAUSE)

    print(f" {count} images sauvegardées pour {species_name}")



# PROGRAMME PRINCIPAL

if __name__ == "__main__":
    for common_name, scientific_name in PENGUINS.items():
        collect_species(common_name, scientific_name)

    print("\n🎉 Dataset créé avec succès !")
