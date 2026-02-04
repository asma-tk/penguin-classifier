import torch
from PIL import Image
from torch.utils.data import Dataset

class PenguinDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return {"pixel_values": self.transform(img), "labels": self.labels[idx]}
        except Exception as e:
            print(f"Erreur sur l'image {self.paths[idx]}: {e}")
            return None