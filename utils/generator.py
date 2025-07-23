import os
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import cv2
import albumentations as A

class FakeImageDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(224, 224), num_classes=2, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.5)
        ])

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            image = self._load_image(self.image_paths[i])
            label = self.labels[i]
            batch_images.append(image)
            batch_labels.append(label)

        return np.array(batch_images), to_categorical(batch_labels, num_classes=self.num_classes)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        if self.augment:
            img = self.transform(image=img)['image']
        
        return img.astype(np.float32) / 255.0
