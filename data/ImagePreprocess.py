from PIL import Image, ImageEnhance

import os
import random


class ImagePreprocess:
    
    def __init__(self, resol, aug_per_image=0):
        self.resol = resol
        self.aug_per_image = aug_per_image

    def horizontal_flip(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def rotate(self, image, angle):
        return image.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    def crop(self, image, ratio):
        w, h = image.size
        dx = int(w * ratio)
        dy = int(h * ratio)
        image = image.crop((dx, dy, w-dx, h-dy))
        return image.resize((w, h), Image.BICUBIC)
    
    def resize(self, image):
        w, h = image.size
        if (w, h) != self.resol:
            image = image.resize(self.resol, Image.BICUBIC)
        return image
    
    def random_augment(self, image):
        if random.random() < 0.5:
            image = self.horizontal_flip(image)
        image = self.rotate(image, random.uniform(-8, 8))
        image = self.crop(image, random.uniform(0, 0.10))

        # Color Jitter
        sat = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        bright = ImageEnhance.Brightness(sat).enhance(random.uniform(0.9, 1.1))
        contrast = ImageEnhance.Contrast(bright).enhance(random.uniform(0.9, 1.1))
        return contrast

    def process(self, src, dest):
        os.makedirs(dest, exist_ok=True)
        files = [f for f in os.listdir(src) if f.lower().endswith(("png","jpg","jpeg"))]
        for file in files:
            image = Image.open(os.path.join(src, file)).convert("RGBA")
            image = self.resize(image)
            image.save(os.path.join(dest, file))
            base_name, ext = os.path.splitext(file)
            for i in range(self.aug_per_image):
                augmented = self.random_augment(image)
                augmented.save(os.path.join(dest, f"{base_name}_aug{i}{ext}"))
