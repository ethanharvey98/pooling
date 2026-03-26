"""
Inspect downloaded THUMOS14 .npy files to understand shapes, dtypes, and annotation format.
Run after download_thumos14.sh completes.

Usage:
    python scripts/inspect_thumos14.py /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14
"""

import sys
import os
import numpy as np

data_dir = sys.argv[1] if len(sys.argv) > 1 else '/cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14'

print(f"=== Contents of {data_dir} ===")
for f in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, f)
    if f.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        print(f"\n--- {f} ---")
        print(f"  type={type(arr)}, dtype={arr.dtype}, shape={arr.shape}")
        if arr.dtype == object:
            print(f"  first 3 entries:")
            for i in range(min(3, len(arr))):
                val = arr[i]
                if hasattr(val, 'shape'):
                    print(f"    [{i}]: type={type(val).__name__}, shape={val.shape}, dtype={val.dtype}")
                elif isinstance(val, (list, tuple)):
                    print(f"    [{i}]: list, len={len(val)}, first={val[:3] if val else '(empty)'}")
                elif isinstance(val, bytes):
                    print(f"    [{i}]: bytes, val={val.decode('utf-8')}")
                else:
                    print(f"    [{i}]: {type(val).__name__}, val={val}")
        else:
            print(f"  first 3 rows: {arr[:3]}")
    else:
        size = os.path.getsize(path) / (1024*1024)
        print(f"\n--- {f} ({size:.1f} MB) ---")

# Feature statistics
print("\n=== Feature Statistics ===")
features = np.load(f'{data_dir}/THUMOS14-I3D-JOINTFeatures.npy', allow_pickle=True)
subsets = np.load(f'{data_dir}/subset.npy', allow_pickle=True)
subsets = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in subsets])

for split in np.unique(subsets):
    mask = subsets == split
    feat_split = features[mask]
    lengths = [f.shape[0] for f in feat_split]
    print(f"\n{split}: {mask.sum()} videos")
    print(f"  temporal lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"  feature dim: {feat_split[0].shape[1]}")
    print(f"  total snippets: {sum(lengths)}")
