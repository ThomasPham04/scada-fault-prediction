# Stratified Sampling Results - SCADA Fault Prediction

## Improvement Summary

Successfully implemented stratified sampling to address extreme class imbalance.

### Before Stratified Sampling (Uniform stride=10)
- **Total samples:** 119,052
- **Positive samples:** 347 (0.29%)
- **Imbalance ratio:** 1:342
- **Memory usage:** Low

### After Stratified Sampling (stride=3 anomaly, stride=100 normal)
- **Total samples:** 134,719
- **Positive samples:** 691 (0.51%)
- **Imbalance ratio:** 1:194
- **Memory usage:** ~20GB peak (manageable)

## Key Improvements

1. **76% Better Class Balance**
   - Reduced imbalance from 1:342 to 1:194
   - Nearly **2x more fault samples** (347 → 691)

2. **Better Fault Coverage**
   - Anomaly events: stride=3 captures fault patterns every 30 minutes
   - Normal events: stride=100 reduces overwhelming normal samples

3. **Memory Efficient**
   - Preprocessing completed in ~1 minute
   - Peak memory ~20GB (fits in standard systems)

## Expected Impact on Model Performance

### Baseline Model (Logistic Regression)
- **Previous:** 0.7% precision, 1.7% recall, 0.886 AUC
- **Expected:** 1.0-1.5% precision, 3-5% recall, similar AUC
- Random classifier expected precision increased from 0.29% to 0.51%

### LSTM Model
- Better class balance should improve precision significantly
- More positive samples = better learning of fault patterns
- Reduced overfitting risk due to more diverse positive samples

## Data Split Details

| Split | Total | Negative | Positive | % Positive |
|-------|-------|----------|----------|------------|
| Train | 89,164 | 88,704 | 460 | 0.52% |
| Val | 22,527 | 22,412 | 115 | 0.51% |
| Test | 23,028 | 22,912 | 116 | 0.50% |

## Next Steps

1. ✅ Retrain baseline model with new balanced dataset
2. ✅ Retrain LSTM model with new balanced dataset
3. [ ] Compare results with previous unbalanced version
4. [ ] Consider additional techniques:
   - Threshold optimization
   - SMOTE oversampling (if needed)
   - Disk-based processing for stride=1 (maximum coverage)
