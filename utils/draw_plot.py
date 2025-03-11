import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_curve(training, validation, mode):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(training) + 1), training, marker='o', linestyle='-', color='b', label=f'Training {mode}')
    plt.plot(range(1, len(validation) + 1), validation, marker='o', linestyle='-', color='r', label=f'Validation {mode}')
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.title(f'{mode} Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'figures/{mode.lower()}_curve.png')
    
def plot_confusion_matrix(label, pred, label_names):
    cm = confusion_matrix(label, pred)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
   
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'figures/confusion_matrix.png')