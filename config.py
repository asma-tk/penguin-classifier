

import os, torch

# ── Chemins ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset_original", "train", "penguins")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "penguin_model")

# ── Paramètres Modèle ────────────────────────────────────
MODEL_NAME       = "google/vit-base-patch16-224"
BASE_MODEL_FT    = "google/vit-base-patch16-224-in21k"
DEVICE           = 0 if torch.cuda.is_available() else -1
IMAGE_SIZE       = 224
BATCH_SIZE       = 16
NUM_EPOCHS       = 5  # 5 suffisent pour éviter l'overfitting



PENGUINS = {
    "Emperor_Penguin": "Aptenodytes forsteri", "King_Penguin": "Aptenodytes patagonicus",
    "Adelie_Penguin": "Pygoscelis adeliae", "Chinstrap_Penguin": "Pygoscelis antarcticus",
    "Gentoo_Penguin": "Pygoscelis papua", "Little_Penguin": "Eudyptula minor",
    "African_Penguin": "Spheniscus demersus", "Galapagos_Penguin": "Spheniscus mendiculus",
    "Humboldt_Penguin": "Spheniscus humboldti", "Magellanic_Penguin": "Spheniscus magellanicus",
    "Yellow-eyed_Penguin": "Megadyptes antipodes", "Macaroni_Penguin": "Eudyptes chrysolophus"
}
FAMILIES = {
    "Aptenodytes (Empereurs / Royaux)": [
        "Emperor_Penguin", "King_Penguin"
    ],
    "Pygoscelis (Dos blanc)": [
        "Adelie_Penguin", "Chinstrap_Penguin", "Gentoo_Penguin"
    ],
    "Eudyptula (Petits manchots)": [
        "Little_Penguin"
    ],
    "Spheniscus (Zones tempérées)": [
        "African_Penguin", "Galapagos_Penguin", "Humboldt_Penguin", "Magellanic_Penguin"
    ],
    "Megadyptes (Yeux jaunes)": [
        "Yellow-eyed_Penguin"
    ],
    "Eudyptes (À huppe)": [
        "Macaroni_Penguin", "Royal_Penguin", "Northern_Rockhopper_Penguin",
        "Southern_Rockhopper_Penguin", "Fiordland_Penguin",
        "Snares_Penguin", "Erect-crested_Penguin"
    ],
}



ALL_SPECIES_DATA = { **PENGUINS}
MANCHOT_GENERA = {"Aptenodytes", "Pygoscelis", "Eudyptula", "Spheniscus", "Megadyptes", "Eudyptes"}
PINGOUIN_GENERA = {"Alca", "Fratercula", "Cepphus", "Uria"}

def classify_bird(name_from_model: str) -> str:
    sci_name = ALL_SPECIES_DATA.get(name_from_model)
    if not sci_name: return "AUTRE"
    genus = sci_name.split()[0]
    return "MANCHOT / PENGUIN" if genus in MANCHOT_GENERA else "PINGOUIN"
