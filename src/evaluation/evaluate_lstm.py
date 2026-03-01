"""
Evaluate LSTM — evaluation.evaluate_lstm
Full test-set evaluation for the LSTM anomaly detector.
Includes adaptive threshold decision logic and comparison to a naive baseline.

Usage:
    python -m src.evaluation.evaluate_lstm
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR
from training.experiments.threshold_tuning import (
    load_events,
    smooth_mae,
    determine_threshold,
)


# ---------------------------------------------------------------------------
# Train / Val error evaluation (used during training)
# ---------------------------------------------------------------------------

def evaluate_train_val(model, X_train, y_train, X_val, y_val) -> dict:
    """
    Compute MSE and MAE per sample for train and val splits.
    Used to inspect model fit quality after training.

    Returns:
        Dict with 'train' and 'val' error stats
    """
    print("\nEvaluating model on train/val splits...")
    errors = {}
    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
        y_pred = model.predict(X, verbose=0)
        mse_per_sample = np.mean((y - y_pred) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(y - y_pred), axis=1)
        errors[name] = {
            'mse_per_sample': mse_per_sample,
            'mae_per_sample': mae_per_sample,
            'mean_mse': float(np.mean(mse_per_sample)),
            'std_mse':  float(np.std(mse_per_sample)),
            'mean_mae': float(np.mean(mae_per_sample)),
            'std_mae':  float(np.std(mae_per_sample)),
        }
        print(f"  {name.capitalize()}: MSE={errors[name]['mean_mse']:.6f} "
              f"± {errors[name]['std_mse']:.6f}  |  "
              f"MAE={errors[name]['mean_mae']:.6f} ± {errors[name]['std_mae']:.6f}")
    return errors


def plot_error_distributions(errors: dict, save_path: str) -> None:
    """Plot MSE and MAE distributions for train and val splits."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for split in ['train', 'val']:
        mse = errors[split]['mse_per_sample']
        mae = errors[split]['mae_per_sample']
        axes[0].hist(mse, bins=50, alpha=0.6, label=f'{split.capitalize()} (μ={mse.mean():.4f})')
        axes[1].hist(mae, bins=50, alpha=0.6, label=f'{split.capitalize()} (μ={mae.mean():.4f})')

    for ax, xlabel, title in zip(axes, ['MSE', 'MAE'], ['MSE Distribution', 'MAE Distribution']):
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error distributions saved to: {save_path}")
    plt.close()


def save_results(errors: dict, metadata: dict, output_dir: str) -> None:
    """Save train/val error stats and raw arrays to disk."""
    results = {
        'version': 2,
        'model_type': 'LSTM_Prediction_V2',
        'window_size': metadata['window_size'],
        'n_features': metadata['n_features'],
        'split_strategy': metadata['split_strategy'],
        'train_error_stats': {k: errors['train'][k] for k in ('mean_mse', 'std_mse', 'mean_mae', 'std_mae')},
        'val_error_stats':   {k: errors['val'][k]   for k in ('mean_mse', 'std_mse', 'mean_mae', 'std_mae')},
    }
    results_path = os.path.join(output_dir, 'lstm_7day_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    errors_path = os.path.join(output_dir, 'lstm_7day_errors.npz')
    np.savez(
        errors_path,
        train_mse=errors['train']['mse_per_sample'],
        val_mse=errors['val']['mse_per_sample'],
        train_mae=errors['train']['mae_per_sample'],
        val_mae=errors['val']['mae_per_sample'],
    )
    print(f"Error arrays saved to: {errors_path}")


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def _longest_run(mask: np.ndarray) -> int:
    """Return length of the longest consecutive True run in a boolean mask."""
    max_run = run = 0
    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def evaluate_test(
    model,
    test_events: dict,
    upper_threshold: float,
    lower_threshold: float,
    min_recall: float = 0.7,
    smoothing_window: int = 3,
) -> dict:
    """
    Evaluate the LSTM on test events using the adaptive hybrid threshold.

    Decision logic:
      - Short events (<18 samples): detect if severity_score > threshold * 0.9
      - Long events: detect if (longest_anomaly_run >= min_run AND
                                (severity_score > adaptive_threshold OR
                                 outlier_ratio >= min_ratio))
                             OR VERY_HIGH_SEVERITY

    Args:
        model: Trained Keras model
        test_events: Event dict from load_events('test')
        upper_threshold: Global upper threshold from determine_threshold()
        lower_threshold: Lower threshold (usually 0)
        min_recall: Recall constraint (informational only here)
        smoothing_window: MAE smoothing window

    Returns:
        Dict with confusion matrix, metrics, and per-event results
    """
    print(f"\nEvaluating with thresholds: upper={upper_threshold:.6f}, lower={lower_threshold:.6f}")

    TP = FP = TN = FN = 0
    results, plot_records = [], []

    print("\n" + "=" * 140)
    print(f"{'Event':<8} {'True Label':<12} {'Predicted':<12} {'Result':<10} "
          f"{'Mean MAE':<12} {'p95 MAE':<12} {'Max MAE':<12} {'Samples':<10}")
    print("-" * 140)

    for event_id, data in sorted(test_events.items()):
        X, y = data['X'], data['y']
        true_label = data['label']

        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        mae = smooth_mae(mae, window=smoothing_window)

        event_std = np.std(mae)
        adaptive_threshold = upper_threshold + 0.45 * event_std
        event_p90 = np.percentile(mae, 90)
        event_p95 = np.percentile(mae, 95)
        severity_score = 0.6 * event_p95 + 0.4 * event_p90

        outlier_ratio = np.mean(mae > adaptive_threshold)
        min_ratio = max(0.15, 5 / len(mae))
        above_mask = mae > adaptive_threshold
        longest_anomaly_run = _longest_run(above_mask)
        min_run = max(2, int(0.2 * len(mae)))
        VERY_HIGH_SEVERITY = severity_score > (adaptive_threshold * 1.3)
        SHORT_EVENT = len(mae) < 18

        if SHORT_EVENT:
            detected = severity_score > (upper_threshold * 0.9) or VERY_HIGH_SEVERITY
        else:
            detected = (
                (longest_anomaly_run >= min_run and
                 (severity_score > adaptive_threshold or outlier_ratio >= min_ratio))
                or VERY_HIGH_SEVERITY
            )

        mean_mae = float(np.mean(mae))
        max_mae = float(np.max(mae))
        is_anomaly = (true_label == 'anomaly')

        if is_anomaly and detected:
            res_type, symbol = 'TP', '[OK] TP'; TP += 1
        elif not is_anomaly and not detected:
            res_type, symbol = 'TN', '[OK] TN'; TN += 1
        elif not is_anomaly and detected:
            res_type, symbol = 'FP', '[X]  FP'; FP += 1
        else:
            res_type, symbol = 'FN', '[X]  FN'; FN += 1

        print(f"{event_id:<8} {true_label:<12} {'anomaly' if detected else 'normal':<12} {symbol:<10} "
              f"{mean_mae:<12.4f} {event_p95:<12.4f} {max_mae:<12.4f} {len(mae):<10}")

        results.append({
            'event_id': int(event_id), 'true_label': true_label,
            'detected': bool(detected), 'result_type': res_type,
            'mae_mean': mean_mae, 'mae_p95': float(event_p95),
            'mae_max': max_mae, 'n_samples': int(len(mae)),
        })
        plot_records.append({
            'event_id': event_id, 'true_label': true_label,
            'detected': detected, 'p95': float(event_p95),
            'mean': mean_mae, 'adaptive_threshold': float(adaptive_threshold),
        })

    # Metrics
    total = TP + FP + TN + FN
    accuracy  = (TP + TN) / total if total > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 140)
    print(f"\nCONFUSION MATRIX:")
    print(f"                    Predicted Anomaly | Predicted Normal")
    print(f"  Actual Anomaly  |       {TP:3d}        |      {FN:3d}")
    print(f"  Actual Normal   |       {FP:3d}        |      {TN:3d}")
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Accuracy:         {accuracy:.2%}")
    print(f"  Detection Rate:   {recall:.2%}")
    print(f"  False Alarm Rate: {far:.2%}")
    print(f"  Precision:        {precision:.2%}")
    print(f"  F1 Score:         {f1:.4f}")

    # Naive baseline comparison
    naive_path = os.path.join(RESULTS_DIR, 'baselines', 'naive_test_evaluation.json')
    if os.path.exists(naive_path):
        with open(naive_path) as f:
            naive = json.load(f)['metrics']
        print(f"\n{'Metric':<30} {'Naive':^20} {'LSTM':^20} {'Delta':^15}")
        print("-" * 85)
        for label, nb_v, lstm_v in [
            ('Detection Rate', naive['recall'], recall),
            ('Precision',       naive['precision'], precision),
            ('False Alarm Rate',naive['false_alarm_rate'], far),
            ('F1 Score',        naive['f1_score'], f1),
        ]:
            print(f"{label:<30} {nb_v:<20.2%} {lstm_v:<20.2%} {lstm_v - nb_v:+.2%}")

    # Save JSON results
    output_path = os.path.join(RESULTS_DIR, '7day', 'lstm_test_evaluation.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'model': 'LSTM_7day',
            'upper_threshold': float(upper_threshold),
            'metrics': {'accuracy': accuracy, 'recall': recall,
                        'precision': precision, 'far': far, 'f1': f1},
            'confusion_matrix': {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN},
            'events': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Plot
    _plot_event_scores(plot_records, upper_threshold)

    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'far': far, 'f1': f1}


def _plot_event_scores(plot_records: list, upper_threshold: float) -> None:
    """Scatter plot of per-event p95 MAE vs global and adaptive thresholds."""
    event_ids  = [r['event_id']          for r in plot_records]
    p95_vals   = [r['p95']               for r in plot_records]
    adap_vals  = [r['adaptive_threshold']for r in plot_records]
    labels     = [r['true_label']        for r in plot_records]
    colors     = ['red' if l == 'anomaly' else 'blue' for l in labels]

    plt.figure(figsize=(14, 5))
    plt.scatter(event_ids, p95_vals, c=colors, s=90, alpha=0.8, label='_nolegend_')
    plt.plot(event_ids, adap_vals, color='green', linestyle=':', marker='o',
             linewidth=2, label='Adaptive Threshold')
    plt.axhline(upper_threshold, color='black', linestyle='--', linewidth=2,
                label='Global Upper Threshold')
    plt.xticks(event_ids)
    plt.xlabel("Event ID")
    plt.ylabel("p95 MAE")
    plt.title("Event-level p95 MAE vs Thresholds")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Normal',  markerfacecolor='blue', markersize=10),
    ]
    plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from tensorflow import keras
    print("=" * 100)
    print("LSTM (7-day) — Final Test Set Evaluation")
    print("=" * 100)

    model_path = os.path.join(MODELS_DIR, 'lstm_7day.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run: python -m src.training.scripts.train_lstm")
        return

    model = keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")

    train_events = load_events('train')
    val_events   = load_events('val')
    test_events  = load_events('test')

    if not val_events or not test_events:
        print("Error: Missing val or test event files.")
        return

    upper_th, lower_th = determine_threshold(model, val_events, train_events, min_recall=0.7)
    evaluate_test(model, test_events, upper_th, lower_th)


if __name__ == '__main__':
    main()
