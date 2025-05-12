from sklearn import metrics
import matplotlib.pyplot as plt
from pathlib import Path
import logging

def eval_metrics(actual, pred):
    logging.info("Calculating evaluation metrics...")
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    return accuracy, f1, auc

def plot_roc_curve(actual, pred, output_dir: Path):
    logging.info("Plotting ROC curve...")
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "ROC_curve.png")
    plt.close()