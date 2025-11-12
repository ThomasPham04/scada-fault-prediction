import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# ============================================= CONFIG =============================================
MODEL_PATH = 'lstm_scada_model.h5'
BATCHES_DIR = 'batches'
VAL_SPLIT_PATH = 'val_batch_indices.npy'
OUTPUT_DIR = 'prediction_results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================= LOAD MODEL =============================================
print(f"[Predict] Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ============================================= LOAD VALIDATION DATA =============================================
val_batch_ids = np.load(VAL_SPLIT_PATH)
print(f"[Predict] Found {len(val_batch_ids)} validation batches.")

X_list, y_list = [], []
for b_id in val_batch_ids:
    x_file = os.path.join(BATCHES_DIR, f'X_batch_{b_id}.npy')
    y_file = os.path.join(BATCHES_DIR, f'y_batch_{b_id}.npy')
    if not (os.path.exists(x_file) and os.path.exists(y_file)):
        print(f"⚠️  Warning: Batch {b_id} missing, skipping.")
        continue

    Xb = np.load(x_file)
    yb = np.load(y_file)
    if yb.dtype == object:  # convert string labels if needed
        yb = np.array([1 if str(v).lower() == 'anomaly' else 0 for v in yb])
    X_list.append(Xb)
    y_list.append(yb)

# Combine all batches into single arrays
X_test = np.concatenate(X_list, axis=0)
y_test = np.concatenate(y_list, axis=0)
print(f"[Predict] Loaded test set: X={X_test.shape}, y={y_test.shape}")

# ============================================= MAKE PREDICTIONS =============================================
print("[Predict] Running inference...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ============================================= METRICS =============================================
print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Print to console
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nClassification Report:")
report = classification_report(y_test, y_pred, digits=4)
print(report)

# Save metrics to file
with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("SCADA FAULT PREDICTION - EVALUATION METRICS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test samples: {len(y_test)}\n")
    f.write(f"Test batches: {len(val_batch_ids)}\n\n")
    f.write("Summary Metrics:\n")
    f.write("-"*60 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write("-"*60 + "\n")
    f.write(report)
    f.write("\n")

print(f"✅ Metrics saved to: {os.path.join(OUTPUT_DIR, 'metrics.txt')}")

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
print(f"✅ Confusion matrix saved to: {os.path.join(OUTPUT_DIR, 'confusion_matrix.npy')}")

# # Plot and save confusion matrix (DISABLED - no plotting)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
# plt.title("Confusion Matrix - SCADA Fault Prediction")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Confusion matrix saved to: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

# ============================================= SAVE PREDICTIONS =============================================
# Save predictions
predictions_data = np.column_stack([y_test, y_pred, y_pred_prob.flatten()])
np.save(os.path.join(OUTPUT_DIR, 'predictions.npy'), predictions_data)

# Save as CSV for easy viewing
with open(os.path.join(OUTPUT_DIR, 'predictions.csv'), 'w') as f:
    f.write("true_label,predicted_label,probability\n")
    for i in range(len(y_test)):
        f.write(f"{y_test[i]},{y_pred[i]},{y_pred_prob[i][0]:.6f}\n")

print(f"✅ Predictions saved to: {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

# # PLOTTING DISABLED - Uncomment below if you want visualizations
# # Plot and save prediction comparison
# plt.figure(figsize=(14, 6))
# sample_size = min(200, len(y_test))
# x_axis = range(sample_size)
# plt.plot(x_axis, y_test[:sample_size], label="True Label", marker='o', alpha=0.7, linewidth=2, markersize=4)
# plt.plot(x_axis, y_pred[:sample_size], label="Predicted Label", marker='x', alpha=0.7, linewidth=2, markersize=6)
# plt.title(f"Predicted vs Actual Labels (first {sample_size} samples)")
# plt.xlabel("Sample Index")
# plt.ylabel("Label (0=Normal, 1=Anomaly)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Prediction comparison plot saved to: {os.path.join(OUTPUT_DIR, 'predictions_comparison.png')}")

# # Plot and save probability distribution
# plt.figure(figsize=(10, 6))
# plt.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.7, label='Normal (True)', color='blue')
# plt.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.7, label='Anomaly (True)', color='red')
# plt.xlabel('Predicted Probability')
# plt.ylabel('Frequency')
# plt.title('Distribution of Prediction Probabilities by True Label')
# plt.legend()
# plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Probability distribution plot saved to: {os.path.join(OUTPUT_DIR, 'probability_distribution.png')}")

# Save summary statistics
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("PREDICTION SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total test samples: {len(y_test)}\n")
    f.write(f"True Normal samples: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)\n")
    f.write(f"True Anomaly samples: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)\n\n")
    f.write(f"Predicted Normal samples: {np.sum(y_pred == 0)} ({np.sum(y_pred == 0)/len(y_pred)*100:.2f}%)\n")
    f.write(f"Predicted Anomaly samples: {np.sum(y_pred == 1)} ({np.sum(y_pred == 1)/len(y_pred)*100:.2f}%)\n\n")
    f.write(f"Correct predictions: {np.sum(y_test == y_pred)} ({accuracy*100:.2f}%)\n")
    f.write(f"Incorrect predictions: {np.sum(y_test != y_pred)} ({(1-accuracy)*100:.2f}%)\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  True Negatives (TN):  {cm[0][0]}\n")
    f.write(f"  False Positives (FP): {cm[0][1]}\n")
    f.write(f"  False Negatives (FN): {cm[1][0]}\n")
    f.write(f"  True Positives (TP):  {cm[1][1]}\n")

print(f"✅ Summary saved to: {os.path.join(OUTPUT_DIR, 'summary.txt')}")

# ============================================= ADDITIONAL METRICS (NO PLOTTING) =============================================
print("\n[Calculating additional metrics...]")

# Calculate ROC AUC
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_test, y_pred_prob)
avg_precision = average_precision_score(y_test, y_pred_prob)

print(f"✅ ROC AUC Score: {roc_auc:.4f}")
print(f"✅ Average Precision Score: {avg_precision:.4f}")

# # PLOTTING DISABLED - Uncomment if you want ROC curve visualization
# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.figure(figsize=(10, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
# plt.legend(loc="lower right", fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ ROC curve saved to: {os.path.join(OUTPUT_DIR, 'roc_curve.png')}")

# # PLOTTING DISABLED - Precision-Recall Curve
# from sklearn.metrics import precision_recall_curve
# precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
# plt.figure(figsize=(10, 8))
# plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, 
#          label=f'PR curve (AP = {avg_precision:.4f})')
# plt.xlabel('Recall', fontsize=12)
# plt.ylabel('Precision', fontsize=12)
# plt.title('Precision-Recall Curve', fontsize=14)
# plt.legend(loc="upper right", fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Precision-Recall curve saved to: {os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')}")

# Threshold Analysis (calculate only, no plotting)
thresholds_to_test = np.arange(0.1, 1.0, 0.05)
accuracies, precisions, recalls, f1_scores = [], [], [], []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_prob > thresh).astype(int).flatten()
    accuracies.append(accuracy_score(y_test, y_pred_thresh))
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))

# # PLOTTING DISABLED - Threshold Analysis
# plt.figure(figsize=(12, 8))
# plt.plot(thresholds_to_test, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
# plt.plot(thresholds_to_test, precisions, 's-', label='Precision', linewidth=2, markersize=6)
# plt.plot(thresholds_to_test, recalls, '^-', label='Recall', linewidth=2, markersize=6)
# plt.plot(thresholds_to_test, f1_scores, 'd-', label='F1-Score', linewidth=2, markersize=6)
# plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Default Threshold (0.5)')
# plt.xlabel('Classification Threshold', fontsize=12)
# plt.ylabel('Score', fontsize=12)
# plt.title('Metrics vs Classification Threshold', fontsize=14)
# plt.legend(loc='best', fontsize=11)
# plt.grid(True, alpha=0.3)
# plt.xlim([0.1, 0.95])
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Threshold analysis saved to: {os.path.join(OUTPUT_DIR, 'threshold_analysis.png')}")

# Find optimal threshold (maximize F1-score)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_to_test[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"\n[Optimal Threshold Analysis]")
print(f"  Optimal threshold: {optimal_threshold:.2f}")
print(f"  F1-Score at optimal: {optimal_f1:.4f}")
print(f"  Accuracy at optimal: {accuracies[optimal_idx]:.4f}")
print(f"  Precision at optimal: {precisions[optimal_idx]:.4f}")
print(f"  Recall at optimal: {recalls[optimal_idx]:.4f}")

# Calculate error statistics (no plotting)
errors_fp = (y_test == 0) & (y_pred == 1)  # False Positives
errors_fn = (y_test == 1) & (y_pred == 0)  # False Negatives
correct = y_test == y_pred

print(f"\n[Error Statistics]")
print(f"  False Positives: {np.sum(errors_fp)}")
print(f"  False Negatives: {np.sum(errors_fn)}")
print(f"  Correct Predictions: {np.sum(correct)}")

# # PLOTTING DISABLED - Error Distribution
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.hist(y_pred_prob[errors_fp], bins=30, alpha=0.7, color='orange', label='False Positives')
# plt.hist(y_pred_prob[errors_fn], bins=30, alpha=0.7, color='red', label='False Negatives')
# plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
# plt.xlabel('Predicted Probability')
# plt.ylabel('Count')
# plt.title('Error Distribution by Type')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.subplot(1, 2, 2)
# data_to_plot = [
#     y_pred_prob[correct & (y_test == 0)].flatten(),
#     y_pred_prob[correct & (y_test == 1)].flatten(),
#     y_pred_prob[errors_fp].flatten(),
#     y_pred_prob[errors_fn].flatten()
# ]
# labels = ['Correct\nNormal', 'Correct\nAnomaly', 'False\nPositive', 'False\nNegative']
# bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
# for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'orange', 'red']):
#     patch.set_facecolor(color)
# plt.ylabel('Predicted Probability')
# plt.title('Prediction Confidence by Outcome')
# plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
# plt.grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'error_analysis.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Error analysis saved to: {os.path.join(OUTPUT_DIR, 'error_analysis.png')}")

# # PLOTTING DISABLED - Prediction Heatmap
# if len(y_test) >= 1000:
#     sample_indices = np.random.choice(len(y_test), 1000, replace=False)
# else:
#     sample_indices = np.arange(len(y_test))
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# ax1.scatter(sample_indices, y_test[sample_indices], c=y_test[sample_indices], 
#             cmap='RdYlGn_r', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
# ax1.set_ylabel('True Label', fontsize=12)
# ax1.set_title('True Labels vs Predicted Probabilities', fontsize=14)
# ax1.set_yticks([0, 1])
# ax1.set_yticklabels(['Normal', 'Anomaly'])
# ax1.grid(True, alpha=0.3, axis='x')
# scatter = ax2.scatter(sample_indices, y_pred_prob[sample_indices], 
#                      c=y_test[sample_indices], cmap='RdYlGn_r', 
#                      alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
# ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
# ax2.set_xlabel('Sample Index', fontsize=12)
# ax2.set_ylabel('Predicted Probability', fontsize=12)
# ax2.set_ylim([-0.05, 1.05])
# ax2.legend()
# ax2.grid(True, alpha=0.3)
# plt.colorbar(scatter, ax=[ax1, ax2], label='True Label (0=Normal, 1=Anomaly)')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_heatmap.png'), dpi=300, bbox_inches='tight')
# plt.close()
# print(f"✅ Prediction heatmap saved to: {os.path.join(OUTPUT_DIR, 'prediction_heatmap.png')}")

print("\n" + "="*60)
print("✅ PREDICTION AND EVALUATION COMPLETE!")
print("="*60)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\n📄 Files Created:")
print("  - metrics.txt - Detailed performance metrics")
print("  - predictions.csv - All predictions with probabilities")
print("  - predictions.npy - Predictions in numpy format")
print("  - confusion_matrix.npy - Confusion matrix")
print("  - summary.txt - Overall summary statistics")
print("\n💡 Note: Plotting is disabled. Uncomment visualization code if needed.")
