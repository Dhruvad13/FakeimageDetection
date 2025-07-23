import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:

    @staticmethod
    def calculate_basic_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate basic classification metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_per_class': precision_score(y_true, y_pred, average=None),
            'recall_per_class': recall_score(y_true, y_pred, average=None),
            'f1_score_per_class': f1_score(y_true, y_pred, average=None)
        }

        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba[:, 1])
            else:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except:
                    metrics['auc_roc'] = None

        return metrics

    @staticmethod
    def calculate_confusion_matrix_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                'confusion_matrix': cm,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'false_discovery_rate': fp / (fp + tp) if (fp + tp) > 0 else 0
            }
        else:
            metrics = {
                'confusion_matrix': cm
            }

        return metrics

    @staticmethod
    def calculate_model_performance(model, X_test, y_test):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        if len(y_test.shape) > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test

        basic_metrics = MetricsCalculator.calculate_basic_metrics(y_test_labels, y_pred, y_pred_proba)
        cm_metrics = MetricsCalculator.calculate_confusion_matrix_metrics(y_test_labels, y_pred)

        return {**basic_metrics, **cm_metrics}

    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=['Real', 'Fake']):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        report = classification_report(y_true, y_pred, target_names=class_names)
        print("Classification Report:")
        print("=" * 50)
        print(report)
        return report

    @staticmethod
    def calculate_fake_probability(prediction):
        prediction = np.array(prediction)

        if prediction.ndim > 1:
            if prediction.shape[0] == 1:
                fake_prob = prediction[0][1] if prediction.shape[1] > 1 else prediction[0][0]
            else:
                fake_prob = prediction[:, 1] if prediction.shape[1] > 1 else prediction[:, 0]
        else:
            fake_prob = prediction[1] if len(prediction) > 1 else prediction[0]

        return float(fake_prob)

    @staticmethod
    def get_prediction_details(prediction, threshold=0.5):
        fake_prob = MetricsCalculator.calculate_fake_probability(prediction)

        result = {
            'fake_probability': fake_prob,
            'real_probability': 1 - fake_prob,
            'predicted_class': 'Fake' if fake_prob > threshold else 'Real',
            'confidence': max(fake_prob, 1 - fake_prob),
            'threshold': threshold
        }

        if result['confidence'] >= 0.9:
            result['confidence_level'] = 'Very High'
        elif result['confidence'] >= 0.8:
            result['confidence_level'] = 'High'
        elif result['confidence'] >= 0.7:
            result['confidence_level'] = 'Moderate'
        elif result['confidence'] >= 0.6:
            result['confidence_level'] = 'Low'
        else:
            result['confidence_level'] = 'Very Low'

        return result

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba):
        y_true = np.array(y_true)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred_proba.shape[1] == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.show()

            return auc_score, fpr, tpr, thresholds

    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba):
        y_true = np.array(y_true)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if y_pred_proba.shape[1] == 2:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
            ap_score = average_precision_score(y_true, y_pred_proba[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'PR curve (AP = {ap_score:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.show()

            return ap_score, precision, recall, thresholds

    @staticmethod
    def evaluate_threshold_performance(y_true, y_pred_proba, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        y_true = np.array(y_true)

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
            metrics = MetricsCalculator.calculate_basic_metrics(y_true, y_pred)
            metrics['threshold'] = threshold
            results.append(metrics)

        return results

    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba, metric='f1_score'):
        threshold_results = MetricsCalculator.evaluate_threshold_performance(y_true, y_pred_proba)
        best_result = max(threshold_results, key=lambda x: x[metric])
        return best_result['threshold'], best_result[metric]

    @staticmethod
    def create_metrics_summary(metrics_dict):
        summary = "\n" + "=" * 60 + "\n"
        summary += "                    MODEL PERFORMANCE SUMMARY\n"
        summary += "=" * 60 + "\n"
        summary += f"Accuracy:           {metrics_dict.get('accuracy', 0):.4f}\n"
        summary += f"Precision:          {metrics_dict.get('precision', 0):.4f}\n"
        summary += f"Recall:             {metrics_dict.get('recall', 0):.4f}\n"
        summary += f"F1-Score:           {metrics_dict.get('f1_score', 0):.4f}\n"

        if 'auc_roc' in metrics_dict and metrics_dict['auc_roc'] is not None:
            summary += f"AUC-ROC:            {metrics_dict['auc_roc']:.4f}\n"
        if 'auc_pr' in metrics_dict and metrics_dict['auc_pr'] is not None:
            summary += f"AUC-PR:             {metrics_dict['auc_pr']:.4f}\n"

        if 'sensitivity' in metrics_dict:
            summary += "\n" + "-" * 60 + "\n"
            summary += "                   DETAILED METRICS\n"
            summary += "-" * 60 + "\n"
            summary += f"Sensitivity (TPR):  {metrics_dict['sensitivity']:.4f}\n"
            summary += f"Specificity (TNR):  {metrics_dict['specificity']:.4f}\n"
            summary += f"False Positive Rate: {metrics_dict['false_positive_rate']:.4f}\n"
            summary += f"False Negative Rate: {metrics_dict['false_negative_rate']:.4f}\n"

        summary += "=" * 60 + "\n"
        return summary

    @staticmethod
    def save_metrics_to_file(metrics_dict, filename='model_metrics.txt'):
        summary = MetricsCalculator.create_metrics_summary(metrics_dict)
        with open(filename, 'w') as f:
            f.write(summary)
            f.write("\nRAW METRICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics_dict.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value}\n")
        print(f"Metrics saved to {filename}")


# ✅ Utility functions
def quick_evaluate(y_true, y_pred, y_pred_proba=None, class_names=['Real', 'Fake']):
    metrics = MetricsCalculator.calculate_basic_metrics(y_true, y_pred, y_pred_proba)
    print(MetricsCalculator.create_metrics_summary(metrics))
    MetricsCalculator.print_classification_report(y_true, y_pred, class_names)
    return metrics


def calculate_fake_confidence(prediction, return_details=False):
    if return_details:
        return MetricsCalculator.get_prediction_details(prediction)
    else:
        return MetricsCalculator.calculate_fake_probability(prediction)
