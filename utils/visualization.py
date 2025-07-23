import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        if 'lr' in history:
            axes[1, 0].plot(history['lr'], label='Learning Rate', color='orange', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')

        if 'precision' in history and 'recall' in history:
            axes[1, 1].plot(history['precision'], label='Precision', linewidth=2)
            axes[1, 1].plot(history['recall'], label='Recall', linewidth=2)
            axes[1, 1].set_title('Precision and Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Real', 'Fake']):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        return cm

    def plot_classification_report(self, y_true, y_pred, class_names=['Real', 'Fake']):
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        df = pd.DataFrame(report).iloc[:-1, :-1].T
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title('Classification Report Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.show()
        return report

    def create_prediction_plot(self, predictions):
        labels = list(predictions.keys())
        values = [float(pred) * 100 for pred in predictions.values()]
        fig = go.Figure([go.Bar(x=labels, y=values, marker_color='lightblue')])
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Model",
            yaxis_title="Confidence (%)",
            template="plotly_dark"
        )
        return fig

    def create_confidence_gauge(self, fake_percentage):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fake_percentage,
            title={'text': "Fake Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "crimson" if fake_percentage > 50 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        return fig

# ✅ Export individual functions for use in app.py
visualizer = Visualizer()
create_prediction_plot = visualizer.create_prediction_plot
create_confidence_gauge = visualizer.create_confidence_gauge
