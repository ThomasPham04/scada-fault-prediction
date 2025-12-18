"""Check which events have prediction data and why some are excluded."""
import pandas as pd
import os

event_info = pd.read_csv('Dataset/raw/Wind Farm A/event_info.csv', sep=';')
event_ids = event_info['event_id'].tolist()

print(f"Total events in dataset: {len(event_ids)}\n")

events_with_pred = []
events_without_pred = []
events_too_short = []

for event_id in event_ids:
    event_label = event_info[event_info['event_id'] == event_id]['event_label'].values[0]
    
    # Load event
    df = pd.read_csv(f'Dataset/raw/Wind Farm A/datasets/{event_id}.csv', sep=';')
    
    # Check if has prediction data
    has_pred = 'prediction' in df['train_test'].values
    
    if not has_pred:
        events_without_pred.append((event_id, event_label))
    else:
        pred_count = (df['train_test'] == 'prediction').sum()
        
        # Check if long enough (need window_size + 1 = 2017)
        if pred_count >= 2017:
            events_with_pred.append((event_id, event_label, pred_count))
        else:
            events_too_short.append((event_id, event_label, pred_count))

print("=" * 70)
print(f"✅ Events WITH sufficient prediction data: {len(events_with_pred)}")
print("=" * 70)
for eid, label, count in events_with_pred:
    print(f"  Event {eid:3d} ({label:7s}): {count:5d} timesteps")

print(f"\n" + "=" * 70)
print(f"❌ Events WITHOUT prediction data: {len(events_without_pred)}")
print("=" * 70)
for eid, label in events_without_pred:
    print(f"  Event {eid:3d} ({label:7s})")

print(f"\n" + "=" * 70)
print(f"⚠️  Events with TOO SHORT prediction data: {len(events_too_short)}")
print("=" * 70)
for eid, label, count in events_too_short:
    print(f"  Event {eid:3d} ({label:7s}): {count:5d} timesteps (need 2017)")

print(f"\n{'='*70}")
print(f"SUMMARY:")
print(f"  Total events: {len(event_ids)}")
print(f"  Usable for test: {len(events_with_pred)} ({len(events_with_pred)/len(event_ids)*100:.1f}%)")
print(f"  No prediction data: {len(events_without_pred)}")
print(f"  Prediction too short: {len(events_too_short)}")
print(f"{'='*70}")
