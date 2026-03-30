# 🐧 Penguin Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)

**AI-powered penguin species identification from images**

</div>

---

## ✨ Overview

`penguin-classifier` is an end-to-end computer vision project that:

- collects penguin images from iNaturalist,
- fine-tunes a Vision Transformer (ViT),
- serves predictions through a FastAPI backend,
- and provides a clean Streamlit interface for interactive use.

The current setup supports **12 penguin species** defined in `config.py`.

---

## 🧠 Species Covered

- Emperor Penguin
- King Penguin
- Adelie Penguin
- Chinstrap Penguin
- Gentoo Penguin
- Little Penguin
- African Penguin
- Galapagos Penguin
- Humboldt Penguin
- Magellanic Penguin
- Yellow-eyed Penguin
- Macaroni Penguin

---

## 🏗️ Project Structure

```text
pinguin/
├── config.py              # Paths, model settings, species map
├── penguins.py            # iNaturalist data collection script
├── train.py               # ViT training pipeline (Transformers Trainer)
├── main.py                # FastAPI inference API
├── streamlit_app.py       # Streamlit frontend
├── style.css              # UI styling for Streamlit
├── Requirements.TXT       # Python dependencies
├── dataset_original/      # Training images
└── penguin_model/         # Fine-tuned model + checkpoints
```

---

## ⚙️ Installation

> macOS / Linux (zsh)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r Requirements.TXT
```

---

## 🚀 Quick Start

### 1) (Optional) Collect dataset images

```bash
python penguins.py
```

### 2) Train / fine-tune model

```bash
python train.py
```

### 3) Start API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 🔌 API Endpoints

### Health check

```http
GET /api/health
```

Response:

```json
{"status":"online"}
```

### Image classification

```http
POST /api/identify/image
Content-Type: multipart/form-data
```

Example:

```bash
curl -X POST "http://localhost:8000/api/identify/image" \
  -F "file=@/path/to/penguin.jpg"
```

---

## 🧪 Model Notes

- Base model: `google/vit-base-patch16-224-in21k`
- Input size: `224x224`
- Training uses data augmentation + validation split
- Model artifacts are saved under `penguin_model/`

---

## 🎨 Interface

The Streamlit app includes:

- drag-and-drop image upload,
- live API health indicator,
- top prediction confidence bar,
- additional candidate species suggestions.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **ML:** PyTorch, Transformers, torchvision, evaluate
- **Data:** iNaturalist API + PIL image processing

---

## 📌 Future Improvements

- Add evaluation report (`accuracy`, confusion matrix, per-class metrics)
- Export model card and training logs
- Add Docker setup for one-command deployment
- Add CI workflow for lint/test checks

---

## 👤 Author

Built by [@asma-tk](https://github.com/asma-tk)

If you want, I can also add a French version (`README.fr.md`) and a few screenshots/gifs section for a more portfolio-style presentation.
