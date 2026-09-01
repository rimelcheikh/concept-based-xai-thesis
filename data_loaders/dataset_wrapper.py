import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomConceptDataset(Dataset):
    def __init__(self, images, labels, concepts, transform=None):
        self.images = images
        self.labels = labels
        self.concepts = concepts
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        concept = torch.tensor(self.concepts[idx], dtype=torch.float)
        return image, label, concept
