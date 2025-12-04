from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import os


class ImageDataset(Dataset):

    def __init__(self, path, img_size):
        self.path = path
        self.files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
