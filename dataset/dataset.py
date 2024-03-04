import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json

class EmoSetDataset(Dataset):
    def __init__(self, data_root, phase='train', transform=None):
        self.data_root = data_root
        self.phase = phase
        self.transform = transform

        with open(os.path.join(data_root, f'{phase}.json'), 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emotion, image_id, image_path, annotation_path = self.data[idx]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, emotion

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = EmoSetDataset(data_root='path/to/dataset', phase='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
