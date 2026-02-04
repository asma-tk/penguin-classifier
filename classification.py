# classification.py ─── classifie dataset_original/ → CSV
#
# Utilisation :  python classification.py
#   → génère classification_results.csv

import os
import csv
from transformers import pipeline

from config import DATASET_DIR, CSV_OUTPUT, MODEL_NAME, DEVICE, DTYPE

# 1️ Pipeline (même config que dans l'API)
clf = pipeline(
    task="image-classification",
    model=MODEL_NAME,
    dtype=DTYPE,
    device=DEVICE,          # 0 = GPU, -1 = CPU
)

# 2️ Collecter tous les chemins d'images
image_paths = []
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images.")

# 3️ Classifier chaque image
results = []
for i, img_path in enumerate(image_paths, 1):
    pred = clf(img_path)[0]                                  # top-1
    folder_name = os.path.basename(os.path.dirname(img_path))
    results.append({
        "folder":     folder_name,
        "image_path": img_path,
        "label":      pred["label"],
        "score":      pred["score"],
    })
    if i % 50 == 0:
        print(f"  {i}/{len(image_paths)} images traitées…")

# 4️ Sauvegarder en CSV
with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["folder", "image_path", "label", "score"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"  Done! Results saved in {CSV_OUTPUT}")