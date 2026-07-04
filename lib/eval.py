import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
from pathlib import Path
from lib.logging_setup import logger

class Evaluator:
    def __init__(self, output_dir="docs/assets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, classes=None):
        if classes is None:
            classes = [str(i) for i in sorted(list(set(y_true) | set(y_pred)))]
            
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix",
               ylabel="True label",
               xlabel="Predicted label")
               
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                        
        fig.tight_layout()
        plot_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plot_path}")
        return plot_path

    def compute_metrics(self, y_true, y_pred, y_prob=None, num_classes=6):
        """
        Computes precision, recall, F1, and Cohen's Kappa.
        """
        from sklearn.metrics import cohen_kappa_score
        
        report = classification_report(
            y_true, y_pred, 
            labels=list(range(num_classes)), 
            output_dict=True, 
            zero_division=0
        )
        
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        
        # Calculate ROC/AUC for each class one-vs-rest
        auc_scores = {}
        if y_prob is not None:
            for c in range(num_classes):
                try:
                    # Binarize labels
                    y_true_bin = (np.array(y_true) == c).astype(int)
                    if len(np.unique(y_true_bin)) > 1:
                        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, c])
                        auc_scores[f"class_{c}"] = auc(fpr, tpr)
                    else:
                        auc_scores[f"class_{c}"] = 0.5
                except Exception:
                    auc_scores[f"class_{c}"] = 0.5

        metrics_summary = {
            "qwk": kappa,
            "report": report,
            "auc": auc_scores
        }
        
        # Save summary to logs
        logger.info(f"Evaluation Metrics - Quadratic Weighted Kappa (QWK): {kappa:.4f}")
        return metrics_summary
