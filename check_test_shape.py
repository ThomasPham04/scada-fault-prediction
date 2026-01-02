import numpy as np
import os

nbm_dir = r'D:\Final Project\scada-fault-prediction\Dataset\processed\Wind Farm A\NBM_7day'
test_dir = os.path.join(nbm_dir, 'test_by_event')

files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
print(f'Total test events: {len(files)}')
print()

# Check first 3 events
for i, filename in enumerate(files[:3]):
    data = np.load(os.path.join(test_dir, filename), allow_pickle=True)
    print(f'{filename}:')
    print(f'  X shape: {data["X"].shape}')
    print(f'  Window size: {data["X"].shape[1]} timesteps')
    print(f'  Label: {data["label"]}')
    print()
