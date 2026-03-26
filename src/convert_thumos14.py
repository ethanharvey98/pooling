"""
Convert THUMOS14 P-MIL .npy files to pipeline .pth format.

Usage (binary - any action present):
    python src/convert_thumos14.py \
        --data_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14 \
        --output_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14_encoded \
        --mode binary --seed 1001

Usage (per-class - one binary task per action class):
    python src/convert_thumos14.py \
        --data_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14 \
        --output_dir /cluster/tufts/hugheslab/dloevl01/datasets/THUMOS14_encoded \
        --mode per_class --class_index 0 --seed 1001
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch


def load_raw_data(data_dir):
    """Load all .npy files from the P-MIL download."""
    features = np.load(f'{data_dir}/THUMOS14-I3D-JOINTFeatures.npy', allow_pickle=True)
    videonames = np.load(f'{data_dir}/videoname.npy', allow_pickle=True)
    subsets = np.load(f'{data_dir}/subset.npy', allow_pickle=True)
    labels_all = np.load(f'{data_dir}/labels_all.npy', allow_pickle=True)
    classlist = np.load(f'{data_dir}/classlist.npy', allow_pickle=True)

    # Decode bytes to strings if needed
    videonames = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in videonames])
    subsets = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in subsets])
    classlist = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in classlist]

    # Convert string labels to multi-hot
    num_classes = len(classlist)
    multi_hot = np.zeros((len(labels_all), num_classes), dtype=np.float32)
    for i, label_list in enumerate(labels_all):
        for label in label_list:
            label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
            if label_str in classlist:
                multi_hot[i, classlist.index(label_str)] = 1.0

    # Load temporal annotations (for test set instance labels)
    segments = np.load(f'{data_dir}/segments.npy', allow_pickle=True)
    seg_labels = np.load(f'{data_dir}/labels.npy', allow_pickle=True)

    return features, videonames, subsets, multi_hot, classlist, segments, seg_labels


def make_bag_label(multi_hot, mode, class_index):
    """Convert multi-hot to binary bag label."""
    if mode == 'binary':
        return (multi_hot.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    else:
        return multi_hot[:, class_index].astype(np.float32).reshape(-1, 1)


def make_instance_labels(features, videonames, segments, seg_labels, classlist, indices, mode, class_index):
    """Convert temporal segment annotations to per-snippet binary labels."""
    lengths_y = []

    for idx in indices:
        T = features[idx].shape[0]
        snippet_labels = np.zeros(T, dtype=int)

        # Find segments belonging to this video
        video_name = videonames[idx]
        for seg, seg_label in zip(segments, seg_labels):
            seg_label_str = seg_label.decode('utf-8') if isinstance(seg_label, bytes) else str(seg_label)

            # Check if this segment belongs to this video (segments may be stored differently)
            # The exact format depends on the download - segments may be per-video or global
            # We'll handle both cases in the actual implementation after inspecting the data
            pass

        lengths_y.append(snippet_labels.tolist())

    return lengths_y


def build_split(features, multi_hot, indices, mode, class_index):
    """Build X, lengths, y tensors for a split."""
    X_list, lengths, y_list = [], [], []

    for idx in indices:
        feat = features[idx]
        X_list.append(torch.tensor(feat, dtype=torch.float32))
        lengths.append(feat.shape[0])
        label = make_bag_label(multi_hot[idx:idx+1], mode, class_index)
        y_list.append(torch.tensor(label))

    X = torch.cat(X_list)
    y = torch.cat(y_list)
    return X, tuple(lengths), y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert THUMOS14 to pipeline .pth format')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='per_class', choices=['binary', 'per_class'])
    parser.add_argument('--class_index', type=int, default=None,
                        help='Class index (0-19) for per_class mode. If omitted, generates all 20.')
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1001)
    args = parser.parse_args()

    features, videonames, subsets, multi_hot, classlist, segments, seg_labels = load_raw_data(args.data_dir)

    print(f"Loaded {len(features)} videos, {len(classlist)} classes")
    print(f"Classes: {classlist}")
    print(f"Feature dims: {features[0].shape}")

    # Split indices
    train_val_mask = subsets == 'validation'
    test_mask = subsets == 'test'
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Determine which class indices to process
    if args.mode == 'binary':
        class_indices = [None]
    elif args.class_index is not None:
        class_indices = [args.class_index]
    else:
        class_indices = list(range(len(classlist)))

    for ci in class_indices:
        # Stratified train/val split
        bag_labels = make_bag_label(multi_hot[train_val_indices], args.mode, ci).flatten()
        # Only stratify if both classes are present
        stratify = bag_labels if len(np.unique(bag_labels)) > 1 else None
        tv_local_train, tv_local_val = train_test_split(
            np.arange(len(train_val_indices)),
            test_size=args.val_fraction,
            random_state=args.seed,
            stratify=stratify,
        )
        train_indices = train_val_indices[tv_local_train]
        val_indices = train_val_indices[tv_local_val]

        # Build splits
        train_X, train_lengths, train_y = build_split(features, multi_hot, train_indices, args.mode, ci)
        val_X, val_lengths, val_y = build_split(features, multi_hot, val_indices, args.mode, ci)
        test_X, test_lengths, test_y = build_split(features, multi_hot, test_indices, args.mode, ci)

        # Output directory
        if args.mode == 'binary':
            out_dir = f'{args.output_dir}/binary/seed={args.seed}'
        else:
            out_dir = f'{args.output_dir}/class={ci}_{classlist[ci]}/seed={args.seed}'
        os.makedirs(out_dir, exist_ok=True)

        # Save splits
        torch.save({'X': train_X, 'lengths': train_lengths, 'y': train_y}, f'{out_dir}/train.pth')
        torch.save({'X': val_X, 'lengths': val_lengths, 'y': val_y}, f'{out_dir}/val.pth')
        torch.save({'X': test_X, 'lengths': test_lengths, 'y': test_y}, f'{out_dir}/test.pth')

        label_name = 'binary' if args.mode == 'binary' else f'{ci}_{classlist[ci]}'
        pos_train = train_y.sum().item()
        pos_test = test_y.sum().item()
        print(f"[{label_name}] train={len(train_lengths)} ({pos_train:.0f}+), "
              f"val={len(val_lengths)}, test={len(test_lengths)} ({pos_test:.0f}+)")
        print(f"  X dims: train={train_X.shape}, val={val_X.shape}, test={test_X.shape}")
        print(f"  Saved to {out_dir}")

    # NOTE: Instance-level labels (lengths_y) for test set will be added after
    # inspecting the exact format of segments.npy and labels.npy on the HPC.
    # The temporal annotation → per-snippet label conversion depends on knowing
    # the snippet duration, which we need to verify from the downloaded data.
    print("\nNOTE: Run inspect_thumos14.py after download to verify annotation format,")
    print("then update this script to generate instance-level labels (lengths_y).")
