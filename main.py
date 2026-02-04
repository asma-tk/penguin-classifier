from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io, os
from transformers import pipeline
from config import *

app = FastAPI(title="API Spécialiste Pingouins 🦅")
clf = None

@app.on_event("startup")
async def load_model():
    global clf
    path = MODEL_SAVE_DIR if os.path.exists(MODEL_SAVE_DIR) else MODEL_NAME
    clf = pipeline("image-classification", model=path, device=DEVICE)
    print(f"✅ Modèle Pingouin chargé depuis : {path}")

@app.post("/api/identify/image")
async def identify(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        preds = clf(img)
        
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

@app.get("/api/model/info")
async def model_info():
    return {
        "success": True, 
        "num_species": len(PENGUINS),
        "type": "Identification Pingouins Arctiques"
    }

@app.get("/api/health")
async def health():
    return {"status": "online"}