import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json

class EmoSetDataset(Dataset):
    def __init__(self, data_root, phase='train', num_emotion_classes=8, transform=None):
        self.data_root = data_root
        self.phase = phase
        self.num_emotion_classes = num_emotion_classes
        self.transform = transform or self.default_transforms()

        # Load metadata
        self.info = json.load(open(os.path.join(data_root, 'info.json')))
        self.data = json.load(open(os.path.join(data_root, f'{phase}.json')))

    def default_transforms(self):
        if self.phase in ['train']:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure 'item' is accessed as a dictionary
        if isinstance(item, list):
            emotion, image_id = item[0], item[1]
        elif isinstance(item, dict):
            emotion, image_id = item['emotion'], item['image_id']
        else:
            raise TypeError("Dataset item is neither a list nor a dictionary")

        img_path = os.path.join(self.data_root, f"{image_id}")
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        output = {
            'image_id': image_id,
            'image': img,
            'emotion_label_idx': torch.tensor(self.info['label2idx'][emotion]),
        }

        return output

# Example usage
if __name__ == "__main__":
    data_root = '../dataset/EmoSet-118K'
    dataset = EmoSetDataset(data_root=data_root, phase='train', num_emotion_classes=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for data in dataloader:
        print(data['image'].shape, data['emotion_label_idx'])
        # Additional attributes can be accessed similarly
        break
