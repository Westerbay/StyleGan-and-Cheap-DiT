from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import os


class ImageDataset(Dataset):

    def __init__(self, path, img_size):
        self.path = path
        self.files = []
        exts = (".jpg", ".jpeg", ".png", ".webp")
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(exts):
                    self.files.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
