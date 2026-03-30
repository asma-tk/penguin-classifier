from fastapi import FastAPI, UploadFile, File  #main.py
from PIL import Image
import io, os
from transformers import pipeline  #bib hugging face pour utiliser le model
from config import *

app = FastAPI(title="API Spécialiste Pingouins 🦅")
clf = None  #variable globale qui stockera le modèle de classification c’est lui qui sert à la prédiction finale

@app.on_event("startup")
async def load_model():   #plusieure requetes en paralelle
    global clf
    path = MODEL_SAVE_DIR if os.path.exists(MODEL_SAVE_DIR) else MODEL_NAME
    clf = pipeline("image-classification", model=path, device=DEVICE) #crée le pipeline de prédiction
    print(f" Modèle Pingouin chargé depuis : {path}")

@app.post("/api/identify/image")  #  route principale 
async def identify(file: UploadFile = File(...)):    # ... signifient "requis
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        preds = clf(img) #le ppl retourne les top prédictions,
        
        results = []
        for p in preds:   
            label = p["label"]
            results.append({
                "species": label.replace("_", " "),
                "scientific": ALL_SPECIES_DATA.get(label, "Inconnu"),
                "probability": round(p["score"] * 100, 2)
            })
        return {"success": True, "predictions": results}
    except Exception as e: 
        return {"success": False, "error": str(e)}


@app.get("/api/health")
async def health():
    return {"status": "online"}