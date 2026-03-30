import os, torch  #config.py #ml bib

# ── Chemins ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset_original", "train", "penguins")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "penguin_model")

# ── Paramètres Modèle ────────────────────────────────────
MODEL_NAME       = "google/vit-base-patch16-224"
BASE_MODEL_FT    = "google/vit-base-patch16-224-in21k"
DEVICE           = 0 if torch.cuda.is_available() else -1. #0 = premier GPU
IMAGE_SIZE       = 224
BATCH_SIZE       = 16                                     #16 × 16 pixels
NUM_EPOCHS       = 30  # suffisant pour éviter l'overfitting

# ── Espèces “fiables” pour dataset réaliste ──────────────
PENGUINS = {
    "Emperor_Penguin": "Aptenodytes forsteri",
    "King_Penguin": "Aptenodytes patagonicus",
    "Adelie_Penguin": "Pygoscelis adeliae",
    "Chinstrap_Penguin": "Pygoscelis antarcticus",
    "Gentoo_Penguin": "Pygoscelis papua",
    "Little_Penguin": "Eudyptula minor",
    "African_Penguin": "Spheniscus demersus",
    "Galapagos_Penguin": "Spheniscus mendiculus",
    "Humboldt_Penguin": "Spheniscus humboldti",
    "Magellanic_Penguin": "Spheniscus magellanicus",
    "Yellow-eyed_Penguin": "Megadyptes antipodes",
    "Macaroni_Penguin": "Eudyptes chrysolophus"
}



ALL_SPECIES_DATA = { **PENGUINS } #Crée une copie 



# ── Type de calcul ──────────────────────────────────────
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32  #pipeline(..., dtype=DTYPE)
