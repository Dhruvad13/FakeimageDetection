import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils.generator import FakeImageDataGenerator

class DataPreprocessor:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        ])

    def load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(self.img_size)
            image_array = np.array(image)
            image_array = image_array.astype(np.float32) / 255.0
            return image_array
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def load_dataset(self, data_dir):
        """Load dataset from directory structure"""
        images, labels = [], []
        
        real_dir = os.path.join(data_dir, 'real')
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image = self.load_image(os.path.join(real_dir, filename))
                    if image is not None:
                        images.append(image)
                        labels.append(0)

        fake_dir = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image = self.load_image(os.path.join(fake_dir, filename))
                    if image is not None:
                        images.append(image)
                        labels.append(1)

        return np.array(images), np.array(labels)

    def augment_data(self, images, labels, augment_factor=2):
        """Apply data augmentation"""
        augmented_images, augmented_labels = [], []

        for i in range(len(images)):
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])

            for _ in range(augment_factor):
                img_uint8 = (images[i] * 255).astype(np.uint8)
                augmented = self.transform(image=img_uint8)
                aug_image = augmented['image'].astype(np.float32) / 255.0
                augmented_images.append(aug_image)
                augmented_labels.append(labels[i])

        return np.array(augmented_images), np.array(augmented_labels)

    def prepare_data(self, data_dir, batch_size=32, img_size=(224, 224)):
        """Split data and return generators"""
        image_paths = []
        labels = []

        for label, folder in enumerate(['real', 'fake']):
            class_dir = os.path.join(data_dir, folder)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Split into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, test_size=0.3, stratify=labels, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # ✅ Slice the data for faster training
        X_train = X_train[:2000]
        y_train = y_train[:2000]
        X_val = X_val[:500]
        y_val = y_val[:500]
        X_test = X_test[:500]
        y_test = y_test[:500]

        # Create generators
        train_gen = FakeImageDataGenerator(X_train, y_train, batch_size=batch_size, img_size=img_size, augment=True)
        val_gen = FakeImageDataGenerator(X_val, y_val, batch_size=batch_size, img_size=img_size)
        test_gen = FakeImageDataGenerator(X_test, y_test, batch_size=batch_size, img_size=img_size)

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return train_gen, val_gen, test_gen




def create_sample_data():
    """Create sample data structure"""
    base_dirs = ['data/real', 'data/fake', 'data/test']
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("Sample data directories created!")
    print("Please add your images to:")
    print("- data/real/ (for real images)")
    print("- data/fake/ (for fake/AI-generated images)")
    print("- data/test/ (for test images)")

def preprocess_image(image, img_size=(128, 128)):
    """
    Preprocess a single PIL image: resize, normalize, return as numpy array.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(img_size)
        image_array = np.array(image).astype(np.float32) / 255.0
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


if __name__ == "__main__":
    create_sample_data()
