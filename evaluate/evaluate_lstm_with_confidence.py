"""
NBM LSTM Evaluation with Confidence Scoring System
Provides confidence levels for each anomaly prediction
"""

import os
import sys
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIND_FARM_A_PROCESSED, MODELS_DIR, RESULTS_DIR

def load_test_data():
    """Load test events from NBM 7-day."""
    nbm_dir = os.path.join(WIND_FARM_A_PROCESSED, 'NBM_7day')
    test_dir = os.path.join(nbm_dir, 'test_by_event')
    
    test_events = {}
    for filename in os.listdir(test_dir):
        if filename.endswith('.npz'):
            event_id = int(filename.split('_')[1].split('.')[0])
            data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
            test_events[event_id] = {
                'X': data['X'],
                'y': data['y'],
                'label': str(data['label'])
            }
    return test_events

def get_thresholds_and_stats():
    """Get thresholds and validation statistics."""
    errors_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'nbm_7day_errors.npz')
    data = np.load(errors_path)
    val_mae = data['val_mae']
    
    thresholds = {
        'mean': np.mean(val_mae),
        'std': np.std(val_mae),
        'p95': np.percentile(val_mae, 95),
        'p99': np.percentile(val_mae, 99),
        'mean_plus_2std': np.mean(val_mae) + 2 * np.std(val_mae)
    }
    
    return thresholds, val_mae

def compute_confidence_score(mae_stats, thresholds, val_mae):
    """
    Compute confidence score using multiple methods.
    
    Returns:
        confidence: 0-100 score
        breakdown: dict with individual scores
    """
    
    # Method 1: Distance from threshold (normalized)
    p95_distance = (mae_stats['p95'] - thresholds['p95']) / thresholds['p95']
    mean_distance = (mae_stats['mean'] - thresholds['mean']) / thresholds['mean']
    
    # Convert to 0-100 scale
    # Negative distance = below threshold = low confidence
    # Positive distance = above threshold = higher confidence
    distance_score = min(100, max(0, 50 + p95_distance * 50))
    
    # Method 2: Multi-signal agreement
    signals = {
        'mean_high': mae_stats['mean'] > thresholds['mean_plus_2std'],
        'p95_high': mae_stats['p95'] > thresholds['p95'],
        'p99_high': mae_stats['p99'] > thresholds['p99'],
        'max_extreme': mae_stats['max'] > thresholds['p99'] * 1.5
    }
    agreement_score = sum(signals.values()) / len(signals) * 100
    
    # Method 3: Percentile ranking (how rare is this event?)
    percentile_rank = (mae_stats['p95'] > val_mae).mean() * 100
    rarity_score = min(100, max(0, (percentile_rank - 50) * 2))  # Scale 50-100 to 0-100
    
    # Combined confidence (weighted average)
    confidence = (
        0.4 * distance_score +
        0.3 * agreement_score +
        0.3 * rarity_score
    )
    
    breakdown = {
        'distance_score': distance_score,
        'agreement_score': agreement_score,
        'rarity_score': rarity_score,
        'signals': signals,
        'p95_distance_pct': p95_distance * 100,
        'percentile_rank': percentile_rank
    }
    
    return confidence, breakdown

def classify_confidence(confidence):
    """Classify confidence into categories."""
    if confidence >= 80:
        return 'HIGH', '🔴'
    elif confidence >= 60:
        return 'MEDIUM', '🟡'
    else:
        return 'LOW', '🟢'

def evaluate_with_confidence(model, test_events, thresholds, val_mae):
    """Evaluate all events with confidence scoring."""
    
    results = []
    
    for event_id, data in sorted(test_events.items()):
        X = data['X']
        y = data['y']
        true_label = data['label']
        
        # Predict
        y_pred = model.predict(X, verbose=0, batch_size=256)
        mae = np.mean(np.abs(y - y_pred), axis=1)
        
        # Compute statistics
        mae_stats = {
            'mean': np.mean(mae),
            'p50': np.percentile(mae, 50),
            'p95': np.percentile(mae, 95),
            'p99': np.percentile(mae, 99),
            'max': np.max(mae),
            'std': np.std(mae)
        }
        
        # Decision (using p95 threshold)
        detected = mae_stats['p95'] > thresholds['p95']
        
        # Compute confidence
        confidence, breakdown = compute_confidence_score(mae_stats, thresholds, val_mae)
        conf_level, conf_emoji = classify_confidence(confidence)
        
        # Confusion matrix type
        is_anomaly = (true_label == 'anomaly')
        if is_anomaly and detected:
            result_type = 'TP'
        elif not is_anomaly and not detected:
            result_type = 'TN'
        elif not is_anomaly and detected:
            result_type = 'FP'
        else:
            result_type = 'FN'
        
        results.append({
            'event_id': event_id,
            'true_label': true_label,
            'detected': detected,
            'result_type': result_type,
            'confidence': confidence,
            'confidence_level': conf_level,
            'mae_stats': mae_stats,
            'confidence_breakdown': breakdown
        })
    
    return results

def print_results(results, thresholds):
    """Print results with confidence scoring."""
    
    print("\n" + "=" * 120)
    print("EVENT EVALUATION WITH CONFIDENCE SCORING")
    print("=" * 120)
    print(f"{'Event':<8} {'True':<10} {'Pred':<10} {'Result':<8} {'Confidence':<12} {'Level':<8} {'MAE_p95':<12} {'Signals':<10}")
    print("-" * 120)
    
    for r in results:
        event_id = r['event_id']
        true_label = r['true_label']
        predicted = 'Anomaly' if r['detected'] else 'Normal'
        result_type = r['result_type']
        confidence = r['confidence']
        conf_level = r['confidence_level']
        mae_p95 = r['mae_stats']['p95']
        
        # Count agreements
        signals_agreed = sum(r['confidence_breakdown']['signals'].values())
        signals_total = len(r['confidence_breakdown']['signals'])
        
        # Color based on confidence
        if conf_level == 'HIGH':
            symbol = '[!!!]'
        elif conf_level == 'MEDIUM':
            symbol = '[ ! ]'
        else:
            symbol = '[   ]'
        
        print(f"{event_id:<8} {true_label:<10} {predicted:<10} {result_type:<8} "
              f"{confidence:<12.1f}% {symbol} {conf_level:<8} {mae_p95:<12.4f} {signals_agreed}/{signals_total}")
    
    # Prioritization summary
    print("\n" + "=" * 120)
    print("PRIORITIZATION SUMMARY")
    print("=" * 120)
    
    # Filter detected anomalies
    detected_anomalies = [r for r in results if r['detected']]
    
    # Group by confidence
    high_conf = [r for r in detected_anomalies if r['confidence_level'] == 'HIGH']
    med_conf = [r for r in detected_anomalies if r['confidence_level'] == 'MEDIUM']
    low_conf = [r for r in detected_anomalies if r['confidence_level'] == 'LOW']
    
    print(f"\n🔴 HIGH Confidence Alerts ({len(high_conf)}) - INVESTIGATE IMMEDIATELY:")
    if high_conf:
        for r in sorted(high_conf, key=lambda x: -x['confidence']):
            true_str = "✓" if r['true_label'] == 'anomaly' else "✗ FALSE ALARM"
            print(f"   Event {r['event_id']}: Confidence {r['confidence']:.1f}% - {true_str}")
    else:
        print("   None")
    
    print(f"\n🟡 MEDIUM Confidence Alerts ({len(med_conf)}) - Investigate within 24h:")
    if med_conf:
        for r in sorted(med_conf, key=lambda x: -x['confidence']):
            true_str = "✓" if r['true_label'] == 'anomaly' else "✗ FALSE ALARM"
            print(f"   Event {r['event_id']}: Confidence {r['confidence']:.1f}% - {true_str}")
    else:
        print("   None")
    
    print(f"\n🟢 LOW Confidence Alerts ({len(low_conf)}) - Monitor/Batch review:")
    if low_conf:
        for r in sorted(low_conf, key=lambda x: -x['confidence']):
            true_str = "✓" if r['true_label'] == 'anomaly' else "✗ FALSE ALARM"
            print(f"   Event {r['event_id']}: Confidence {r['confidence']:.1f}% - {true_str}")
    else:
        print("   None")
    
    # Workload reduction analysis
    print("\n" + "=" * 120)
    print("WORKLOAD REDUCTION ANALYSIS")
    print("=" * 120)
    
    total_alerts = len(detected_anomalies)
    true_anomalies_in_high = sum(1 for r in high_conf if r['true_label'] == 'anomaly')
    true_anomalies_total = sum(1 for r in results if r['true_label'] == 'anomaly')
    
    print(f"\nTotal Alerts: {total_alerts}")
    print(f"  HIGH priority: {len(high_conf)} alerts")
    print(f"  MEDIUM priority: {len(med_conf)} alerts")
    print(f"  LOW priority: {len(low_conf)} alerts")
    
    if high_conf:
        print(f"\nIf only investigating HIGH confidence:")
        print(f"  Workload reduction: {total_alerts - len(high_conf)} alerts saved ({(total_alerts - len(high_conf))/total_alerts*100:.1f}%)")
        print(f"  True anomalies caught: {true_anomalies_in_high}/{true_anomalies_total} ({true_anomalies_in_high/true_anomalies_total*100:.1f}%)")
    
    if high_conf or med_conf:
        true_in_high_med = sum(1 for r in high_conf + med_conf if r['true_label'] == 'anomaly')
        print(f"\nIf investigating HIGH + MEDIUM:")
        print(f"  Workload reduction: {total_alerts - len(high_conf) - len(med_conf)} alerts saved")
        print(f"  True anomalies caught: {true_in_high_med}/{true_anomalies_total} ({true_in_high_med/true_anomalies_total*100:.1f}%)")

def main():
    print("=" * 120)
    print("NBM LSTM EVALUATION WITH CONFIDENCE SCORING")
    print("=" * 120)
    
    # Load model
    model_path = os.path.join(MODELS_DIR, 'nbm_lstm_7day.keras')
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load test data
    print("Loading test events...")
    test_events = load_test_data()
    print(f"  Loaded {len(test_events)} test events")
    
    # Get thresholds
    print("\nComputing thresholds and validation statistics...")
    thresholds, val_mae = get_thresholds_and_stats()
    
    print("\nThresholds:")
    print(f"  Mean: {thresholds['mean']:.6f}")
    print(f"  Std: {thresholds['std']:.6f}")
    print(f"  p95: {thresholds['p95']:.6f}")
    print(f"  p99: {thresholds['p99']:.6f}")
    
    # Evaluate with confidence
    print("\nEvaluating events with confidence scoring...")
    results = evaluate_with_confidence(model, test_events, thresholds, val_mae)
    
    # Print results
    print_results(results, thresholds)
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, 'NBM_7day', 'lstm_evaluation_with_confidence.json')
    
    # Convert numpy types for JSON
    results_serializable = []
    for r in results:
        r_copy = {
            'event_id': int(r['event_id']),
            'true_label': r['true_label'],
            'detected': bool(r['detected']),
            'result_type': r['result_type'],
            'confidence': float(r['confidence']),
            'confidence_level': r['confidence_level'],
            'mae_stats': {k: float(v) for k, v in r['mae_stats'].items()},
            'confidence_breakdown': {
                'distance_score': float(r['confidence_breakdown']['distance_score']),
                'agreement_score': float(r['confidence_breakdown']['agreement_score']),
                'rarity_score': float(r['confidence_breakdown']['rarity_score']),
                'signals': {k: bool(v) for k, v in r['confidence_breakdown']['signals'].items()},
                'p95_distance_pct': float(r['confidence_breakdown']['p95_distance_pct']),
                'percentile_rank': float(r['confidence_breakdown']['percentile_rank'])
            }
        }
        results_serializable.append(r_copy)
    
    with open(output_path, 'w') as f:
        json.dump({
            'thresholds': {k: float(v) for k, v in thresholds.items()},
            'results': results_serializable
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print("=" * 120)

if __name__ == "__main__":
    main()
