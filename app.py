import gradio as gr
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('models')
sys.path.append('utils')

from cnn_model import create_cnn_model
from resnet_model import ResNetFakeImageDetector
from utils.data_preprocessing import preprocess_image
from visualization import create_prediction_plot, create_confidence_gauge
from metrics import MetricsCalculator, quick_evaluate, calculate_fake_confidence
from metrics import calculate_fake_confidence



class FakeImageDetector:
    def __init__(self):
        self.device = "cpu"
        self.cnn_model = None
        self.resnet_model = None
        self.load_models()  # This line is causing the error if load_models() is not defined

    def load_models(self):
        try:
            print("Loading models...")

            # ✅ Load Keras CNN model if available
            if os.path.exists("saved_models/cnn_model.h5"):
                self.cnn_model = tf.keras.models.load_model("saved_models/cnn_model.h5")
            else:
                print("CNN model not found, creating new model")
                base_model = tf.keras.applications.ResNet50(
                    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
                )
                self.cnn_model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(2, activation='softmax')
                ])
                self.cnn_model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

            # 🔒 Disable PyTorch loading if not needed
            self.resnet_model = None

        except Exception as e:
            print("Error loading models:", str(e))
            self.cnn_model = None
            self.resnet_model = None




    def predict_image(self, image, model_type='ensemble'):
        try:
            processed_img = preprocess_image(image)
            predictions = {}

            if model_type in ['cnn', 'ensemble'] and self.cnn_model is not None:
                cnn_input = np.expand_dims(processed_img, axis=0)
                cnn_pred = self.cnn_model.predict(cnn_input, verbose=0)[0][0]
                predictions['cnn'] = float(cnn_pred)

            if model_type in ['resnet', 'ensemble'] and self.resnet_model is not None:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                img_rgb = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                resnet_input = transform(img_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    resnet_output = self.resnet_model(resnet_input)
                    resnet_pred = torch.sigmoid(resnet_output).cpu().numpy()[0][0]
                    predictions['resnet'] = float(resnet_pred)

            if model_type == 'ensemble' and len(predictions) > 0:
                ensemble_pred = np.mean(list(predictions.values()))
                predictions['ensemble'] = ensemble_pred

            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'error': str(e)}

    def analyze_image_features(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array

            features = {
                'dimensions': f"{img_array.shape[1]} x {img_array.shape[0]}",
                'channels': img_array.shape[2] if len(img_array.shape) == 3 else 1,
                'file_size': f"{img_array.nbytes / 1024:.1f} KB",
                'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist() if len(img_array.shape) == 3 else [],
                'std_rgb': np.std(img_array, axis=(0, 1)).tolist() if len(img_array.shape) == 3 else [],
                'edge_density': np.sum(cv2.Canny(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_cv, 50, 150) > 0) / (img_array.shape[0] * img_array.shape[1]),
                'texture_variance': cv2.Laplacian(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                'noise_level': np.std(cv2.GaussianBlur(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), (5, 5), 0) - cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY))
            }
            return features
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return {'error': str(e)}
    def predict(self, image):
        """Main prediction method for Gradio"""
        image_array = preprocess_image(image, img_size=(224, 224))  # Resize + normalize
        if image_array is None:
            return "Error loading image", 0.0
        image_array = np.expand_dims(image_array, axis=0)
        preds = self.cnn_model.predict(image_array)
        result = MetricsCalculator.get_prediction_details(preds)
        return result['predicted_class'], result['fake_probability'] * 100

    def launch(self):
        """Start Gradio Web App"""
        interface = gr.Interface(
            fn=self.predict,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Label(num_top_classes=2, label="Prediction"),
                gr.Number(label="Fake Probability (%)")
            ],
            title="Fake Image Detection",
            description="Upload an image to predict whether it's real or AI-generated (fake)."
        )
        interface.launch()
# Gradio functions here (can be added if needed)

if __name__ == "__main__":
    detector = FakeImageDetector()
    detector.launch()  
    model.save('saved_models/cnn_model.h5')
