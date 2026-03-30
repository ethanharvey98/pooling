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


def decode(val):
    return val.decode('latin1') if isinstance(val, bytes) else str(val)


def load_raw_data(data_dir):
    """Load all .npy files from the P-MIL download."""
    # Auto-detect feature file name
    for name in ['Thumos14reduced-I3D-JOINTFeatures.npy', 'THUMOS14-I3D-JOINTFeatures.npy']:
        if os.path.exists(f'{data_dir}/{name}'):
            feat_path = f'{data_dir}/{name}'
            break
    else:
        raise FileNotFoundError("Could not find feature .npy file")

    # Auto-detect annotations directory
    for name in ['Thumos14reduced-Annotations', 'Thumos14-Annotations']:
        if os.path.isdir(f'{data_dir}/{name}'):
            ann_dir = f'{data_dir}/{name}'
            break
    else:
        ann_dir = data_dir  # annotations might be in the same directory

    load = lambda f: np.load(f, allow_pickle=True, encoding='latin1')

    features = load(feat_path)
    videonames = np.array([decode(v) for v in load(f'{ann_dir}/videoname.npy')])
    subsets = np.array([decode(s) for s in load(f'{ann_dir}/subset.npy')])
    labels_all = load(f'{ann_dir}/labels_all.npy')  # per-video: list of class name strings
    classlist = [decode(c) for c in load(f'{ann_dir}/classlist.npy')]
    segments = load(f'{ann_dir}/segments.npy')       # per-video: list of [start, end] pairs
    seg_labels = load(f'{ann_dir}/labels.npy')       # per-video: list of class names per segment
    duration = load(f'{ann_dir}/duration.npy')        # (412, 1) video durations in seconds

    # Convert string labels to multi-hot
    num_classes = len(classlist)
    multi_hot = np.zeros((len(labels_all), num_classes), dtype=np.float32)
    for i, label_list in enumerate(labels_all):
        for label in label_list:
            label_str = decode(label)
            if label_str in classlist:
                multi_hot[i, classlist.index(label_str)] = 1.0

    return features, videonames, subsets, multi_hot, classlist, segments, seg_labels, duration


def make_bag_label(multi_hot, mode, class_index):
    if mode == 'binary':
        return (multi_hot.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    else:
        return multi_hot[:, class_index].astype(np.float32).reshape(-1, 1)


def make_instance_labels(features, segments, seg_labels, duration, classlist, indices, mode, class_index):
    """Convert temporal segment annotations to per-snippet binary labels.

    Each snippet covers duration[i] / T_i seconds. A snippet is positive if its
    center falls within a ground-truth segment of the target class(es).
    """
    lengths_y = []

    for idx in indices:
        T = features[idx].shape[0]
        vid_duration = duration[idx].item()
        snippet_duration = vid_duration / T

        # Snippet center times
        snippet_centers = np.array([(j + 0.5) * snippet_duration for j in range(T)])
        snippet_labels = np.zeros(T, dtype=int)

        vid_segments = segments[idx]  # list of [start, end]
        vid_seg_labels = seg_labels[idx]  # list of class name strings

        for seg, seg_label in zip(vid_segments, vid_seg_labels):
            seg_label_str = decode(seg_label)

            # Check if this segment's class matches the target
            if mode == 'binary':
                matches = True
            else:
                matches = (seg_label_str == classlist[class_index])

            if matches:
                start, end = seg[0], seg[1]
                # Mark snippets whose center falls within [start, end]
                mask = (snippet_centers >= start) & (snippet_centers <= end)
                snippet_labels[mask] = 1

        lengths_y.append(snippet_labels.tolist())

    return lengths_y


def build_split(features, multi_hot, segments, seg_labels, duration, classlist,
                indices, mode, class_index, include_instance_labels=False):
    """Build X, lengths, y (and optionally lengths_y) for a split."""
    X_list, lengths, y_list = [], [], []

    for idx in indices:
        feat = features[idx]
        X_list.append(torch.tensor(feat, dtype=torch.float32))
        lengths.append(feat.shape[0])
        label = make_bag_label(multi_hot[idx:idx+1], mode, class_index)
        y_list.append(torch.tensor(label))

    X = torch.cat(X_list)
    y = torch.cat(y_list)

    result = {'X': X, 'lengths': tuple(lengths), 'y': y}

    if include_instance_labels:
        lengths_y = make_instance_labels(
            features, segments, seg_labels, duration, classlist, indices, mode, class_index
        )
        result['lengths_y'] = lengths_y

    return result


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

    features, videonames, subsets, multi_hot, classlist, segments, seg_labels, duration = load_raw_data(args.data_dir)

    print(f"Loaded {len(features)} videos, {len(classlist)} classes")
    print(f"Classes: {classlist}")
    print(f"Feature dim: {features[0].shape[1]}")

    # Split indices
    train_val_mask = subsets == 'validation'
    test_mask = subsets == 'test'
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    print(f"Train+val: {len(train_val_indices)} videos, Test: {len(test_indices)} videos")

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
        stratify = bag_labels if len(np.unique(bag_labels)) > 1 else None
        tv_local_train, tv_local_val = train_test_split(
            np.arange(len(train_val_indices)),
            test_size=args.val_fraction,
            random_state=args.seed,
            stratify=stratify,
        )
        train_indices = train_val_indices[tv_local_train]
        val_indices = train_val_indices[tv_local_val]

        # Build splits (instance labels for all splits since annotations exist for both)
        train_data = build_split(features, multi_hot, segments, seg_labels, duration, classlist,
                                 train_indices, args.mode, ci, include_instance_labels=True)
        val_data = build_split(features, multi_hot, segments, seg_labels, duration, classlist,
                               val_indices, args.mode, ci, include_instance_labels=True)
        test_data = build_split(features, multi_hot, segments, seg_labels, duration, classlist,
                                test_indices, args.mode, ci, include_instance_labels=True)

        # Output directory
        if args.mode == 'binary':
            out_dir = f'{args.output_dir}/binary/seed={args.seed}'
        else:
            out_dir = f'{args.output_dir}/class={ci}_{classlist[ci]}/seed={args.seed}'
        os.makedirs(out_dir, exist_ok=True)

        torch.save(train_data, f'{out_dir}/train.pth')
        torch.save(val_data, f'{out_dir}/val.pth')
        torch.save(test_data, f'{out_dir}/test.pth')

        label_name = 'binary' if args.mode == 'binary' else f'{ci}_{classlist[ci]}'
        pos_train = train_data['y'].sum().item()
        pos_test = test_data['y'].sum().item()

        # Count positive snippets in test set
        test_pos_snippets = sum(sum(ly) for ly in test_data['lengths_y'])
        test_total_snippets = sum(test_data['lengths'])
        print(f"[{label_name}] train={len(train_data['lengths'])} ({pos_train:.0f}+ bags), "
              f"val={len(val_data['lengths'])}, test={len(test_data['lengths'])} ({pos_test:.0f}+ bags, "
              f"{test_pos_snippets}/{test_total_snippets} positive snippets)")
        print(f"  Saved to {out_dir}")
