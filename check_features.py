"""Check feature engineering results."""
import joblib

metadata = joblib.load('Dataset/processed/Wind Farm A/NBM_v2/nbm_metadata_v2.pkl')

print(f"Total features after engineering: {metadata['n_features']}")
print(f"\nFeature list ({len(metadata['feature_columns'])} features):")
print("="*60)

# Group by type
temp_features = [f for f in metadata['feature_columns'] if any(x in f for x in ['sensor_0', 'sensor_6', 'sensor_7', 'sensor_9', 'sensor_1']) and '_sin' not in f and '_cos' not in f]
angle_sin = [f for f in metadata['feature_columns'] if '_sin' in f]
angle_cos = [f for f in metadata['feature_columns'] if '_cos' in f]

print(f"\nBreakdown:")
print(f"  Sin features: {len(angle_sin)}")
print(f"  Cos features: {len(angle_cos)}")
print(f"  Total angle features: {len(angle_sin) + len(angle_cos)}")
print(f"  Other features: {len(metadata['feature_columns']) - len(angle_sin) - len(angle_cos)}")

print(f"\nFirst 15 features:")
for i, feat in enumerate(metadata['feature_columns'][:15]):
    print(f"  {i+1:2d}. {feat}")

print(f"\nLast 10 features (sin/cos):")
for i, feat in enumerate(metadata['feature_columns'][-10:]):
    print(f"  {len(metadata['feature_columns'])-9+i:2d}. {feat}")
