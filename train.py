import os, numpy as np, torch, evaluate
from PIL import Image
from sklearn.model_selection import train_test_split   #  diviser le dataset en train/validation
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from config import *

class PenguinDataset(torch.utils.data.Dataset):   #: stocke les chemins des image
    def __init__(self, paths, labels, transform):  #Constructeur : reçoit les chemins d’images, labels et transformations
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)   #le nombre total d'images
    def __getitem__(self, idx):  
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img), "labels": self.labels[idx]} #Applique les transformations 

def main():  #trie alphabétiquement s, entraîne le modèle et le sauvegarde
    species_list = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    paths, labels = [], []
    l2i = {name: i for i, name in enumerate(species_list)} #prédit 2 pour une image adelie pinguin
    i2l = {i: name for i, name in enumerate(species_list)}

    for sp in species_list:
        folder = os.path.join(DATASET_DIR, sp)
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.join(folder, f))    
                labels.append(l2i[sp])
      #80% des images pour l'entraînement (~2400 images)20% pour la validation (~600 images)
    train_p, val_p, train_l, val_l = train_test_split(paths, labels, test_size=0.2, stratify=labels)
    proc = AutoImageProcessor.from_pretrained(BASE_MODEL_FT) #classe de Hugging Face pour préparer les images avant de les passer au modèle. redimensionner
    norm = Normalize(mean=proc.image_mean, std=proc.image_std) #PyTorch qui normalise les pixels d’une image.
    
    train_ds = PenguinDataset(train_p, train_l, Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), norm]))
    val_ds = PenguinDataset(val_p, val_l, Compose([RandomResizedCrop(224), ToTensor(), norm]))

    model = AutoModelForImageClassification.from_pretrained(BASE_MODEL_FT, num_labels=len(species_list), id2label=i2l, label2id=l2i, ignore_mismatched_sizes=True)
    #arg # Charger le meilleur modèle à la fin load best model
    args = TrainingArguments(output_dir=MODEL_SAVE_DIR, eval_strategy="epoch", save_strategy="epoch", num_train_epochs=NUM_EPOCHS, load_best_model_at_end=True, remove_unused_columns=False)
    # # Dataset de validation val_ds
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()                              # Lancement de l’entraînement
    trainer.save_model(MODEL_SAVE_DIR)
    proc.save_pretrained(MODEL_SAVE_DIR)

if __name__ == "__main__": main()  #automatiquement. »