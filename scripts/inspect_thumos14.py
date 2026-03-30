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
        arr = np.load(path, allow_pickle=True, encoding='latin1')
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
                    print(f"    [{i}]: bytes, val={val.decode('latin1')}")
                elif isinstance(val, str):
                    print(f"    [{i}]: str, val={val}")
                else:
                    print(f"    [{i}]: {type(val).__name__}, val={repr(val)[:200]}")
        else:
            print(f"  first 3 rows: {arr[:3]}")
    elif os.path.isfile(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"\n--- {f} ({size:.1f} MB) ---")

# Feature statistics
print("\n=== Feature Statistics ===")
# Auto-detect feature file name
feat_file = None
for candidate in ['Thumos14reduced-I3D-JOINTFeatures.npy', 'THUMOS14-I3D-JOINTFeatures.npy']:
    if os.path.exists(f'{data_dir}/{candidate}'):
        feat_file = candidate
        break

if feat_file is None:
    print("Could not find feature file!")
    sys.exit(1)

# Check for annotations subdirectory
ann_dir = None
for candidate in ['Thumos14reduced-Annotations', 'Thumos14-Annotations']:
    if os.path.isdir(f'{data_dir}/{candidate}'):
        ann_dir = f'{data_dir}/{candidate}'
        break

if ann_dir:
    print(f"\n=== Annotations directory: {ann_dir} ===")
    for f in sorted(os.listdir(ann_dir)):
        path = os.path.join(ann_dir, f)
        if f.endswith('.npy'):
            arr = np.load(path, allow_pickle=True, encoding='latin1')
            print(f"\n--- {f} ---")
            print(f"  type={type(arr)}, dtype={arr.dtype}, shape={arr.shape}")
            if arr.dtype == object:
                for i in range(min(3, len(arr))):
                    val = arr[i]
                    if hasattr(val, 'shape'):
                        print(f"    [{i}]: type={type(val).__name__}, shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, (list, tuple)):
                        print(f"    [{i}]: list, len={len(val)}, first={val[:3]}")
                    elif isinstance(val, (bytes, str)):
                        s = val.decode('latin1') if isinstance(val, bytes) else val
                        print(f"    [{i}]: {s}")
                    else:
                        print(f"    [{i}]: {type(val).__name__}, val={repr(val)[:200]}")
            else:
                print(f"  first 5 rows: {arr[:5]}")
        elif os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"\n--- {f} ({size:.1f} MB) ---")

features = np.load(f'{data_dir}/{feat_file}', allow_pickle=True, encoding='latin1')

# Try to find subset file in annotations dir or data dir
subset_path = None
for d in [ann_dir, data_dir]:
    if d and os.path.exists(f'{d}/subset.npy'):
        subset_path = f'{d}/subset.npy'
        break

if subset_path is None:
    print("\nCould not find subset.npy - listing all features")
    lengths = [f.shape[0] for f in features]
    print(f"All: {len(features)} videos")
    print(f"  temporal lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"  feature dim: {features[0].shape[1]}")
    print(f"  total snippets: {sum(lengths)}")
    sys.exit(0)

subsets = np.load(subset_path, allow_pickle=True, encoding='latin1')
subsets = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in subsets])

for split in np.unique(subsets):
    mask = subsets == split
    feat_split = features[mask]
    lengths = [f.shape[0] for f in feat_split]
    print(f"\n{split}: {mask.sum()} videos")
    print(f"  temporal lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"  feature dim: {feat_split[0].shape[1]}")
    print(f"  total snippets: {sum(lengths)}")
