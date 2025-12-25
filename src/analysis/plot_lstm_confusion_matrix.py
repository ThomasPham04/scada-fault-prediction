"""
Plot LSTM Confusion Matrix
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

def main():
    # Load results
    results_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'lstm_test_evaluation.json')
    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)
        
    cm_data = data['confusion_matrix']
    TP = cm_data['TP']
    FN = cm_data['FN']
    FP = cm_data['FP']
    TN = cm_data['TN']
    
    # Construct matrix for display
    #              Predicted Anomaly | Predicted Normal
    # Actual Anomaly      TP                FN
    # Actual Normal       FP                TN
    
    # Note: Typically Confusion Matrix is:
    #                 Pred 0 (Norm)   Pred 1 (Anom)
    # Actual 0 (Norm)      TN              FP
    # Actual 1 (Anom)      FN              TP
    
    # Let's align with standard sklearn confusion_matrix convention (if labels are [0, 1])
    # [[TN, FP],
    #  [FN, TP]]
    
    cm = np.array([[TN, FP], [FN, TP]])
    labels = ['Normal', 'Anomaly']
    
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'NBM LSTM Confusion Matrix\n(Threshold p95)', fontsize=14)
    
    # Add metrics text below
    metrics_text = (
        f"Detection Rate (Recall): {TP/(TP+FN):.1%}\n"
        f"False Alarm Rate: {FP/(FP+TN) if (FP+TN)>0 else 0:.1%}\n"
        f"Precision: {TP/(TP+FP) if (TP+FP)>0 else 0:.1%}\n"
        f"F1 Score: {2*TP/(2*TP+FP+FN):.4f}"
    )
    # plt.figtext(0.5, 0.01, metrics_text, wrap=True, horizontalalignment='center', fontsize=10)
    
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'lstm_confusion_matrix.png')
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Make room for text
    plt.savefig(output_path, dpi=300)
    print(f"Confusion matrix saved to: {output_path}")

if __name__ == "__main__":
    main()
