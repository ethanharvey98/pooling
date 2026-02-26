#!/usr/bin/env python3
"""
Radiomic feature extraction and WMD classification for KPSC MRI data.
Uses preprocessed .npz files with Otsu thresholding for mask generation.

Usage:
    python radiomics_kpsc_threshold.py --numpy_dir /path/to/KPSC_MRI_800_numpy
"""

import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from tqdm import tqdm
import warnings

# Suppress radiomics logging
import logging
logging.getLogger('radiomics').setLevel(logging.ERROR)


def numpy_to_sitk(image_np):
    """Convert numpy array to SimpleITK image."""
    # SimpleITK expects (z, y, x) ordering, numpy is (x, y, z) from .npz
    # The .npz shape is (169, 208, N_slices), so transpose to (N_slices, 208, 169)
    image_transposed = np.transpose(image_np, (2, 1, 0))
    sitk_image = sitk.GetImageFromArray(image_transposed.astype(np.float32))
    return sitk_image


def create_threshold_mask(image_sitk):
    """Create binary mask using Otsu thresholding."""
    # Apply Otsu threshold
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask = otsu_filter.Execute(image_sitk)

    # Cast to integer type for radiomics
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    return mask


def extract_features(image_sitk, mask_sitk, extractor, prefix=''):
    """Extract radiomic features from image with mask."""
    try:
        features = extractor.execute(image_sitk, mask_sitk)
        # Filter to only feature values (not diagnostics)
        feature_dict = {}
        for key, val in features.items():
            if not key.startswith('diagnostics'):
                feature_name = f"{prefix}_{key}" if prefix else key
                feature_dict[feature_name] = float(val)
        return feature_dict
    except Exception as e:
        warnings.warn(f"Feature extraction failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Radiomic features for WMD classification (threshold mask)'
    )
    parser.add_argument(
        '--numpy_dir',
        default='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy',
        help='Path to KPSC_MRI_800_numpy directory'
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help='Path to save extracted features CSV (optional)'
    )
    parser.add_argument(
        '--test_site_ids',
        nargs='+',
        type=int,
        default=[9],
        help='Site IDs for test set'
    )
    parser.add_argument(
        '--train_site_ids',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 6, 7, 8, 10, 11],
        help='Site IDs for training'
    )
    parser.add_argument(
        '--val_site_ids',
        nargs='+',
        type=int,
        default=[5],
        help='Site IDs for validation'
    )
    args = parser.parse_args()

    # Load labels
    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')
    print(f"Loaded {len(labels_df)} subjects from labels.csv")

    # Site-based splitting
    train_df = labels_df[labels_df['SiteID'].isin(args.train_site_ids)]
    val_df = labels_df[labels_df['SiteID'].isin(args.val_site_ids)]
    test_df = labels_df[labels_df['SiteID'].isin(args.test_site_ids)]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Initialize PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()

    # Disable features that require specific image types or are slow
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')

    # Extract features for all subjects
    all_features = []
    all_labels = []
    all_study_ids = []
    all_splits = []

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nExtracting features for {split_name} set...")

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            study_id = row['STUDY_ID']
            label = row['idWMD']
            npz_path = row['path']

            try:
                # Load image data
                data = np.load(npz_path)['arr_0']
                t1 = data[0]  # T1 image
                t2 = data[1]  # T2 image

                # Convert to SimpleITK
                t1_sitk = numpy_to_sitk(t1)
                t2_sitk = numpy_to_sitk(t2)

                # Create masks
                t1_mask = create_threshold_mask(t1_sitk)
                t2_mask = create_threshold_mask(t2_sitk)

                # Extract features
                t1_features = extract_features(t1_sitk, t1_mask, extractor, prefix='T1')
                t2_features = extract_features(t2_sitk, t2_mask, extractor, prefix='T2')

                if t1_features is None or t2_features is None:
                    warnings.warn(f"Skipping {study_id} due to extraction failure")
                    continue

                # Combine features
                combined_features = {**t1_features, **t2_features}
                all_features.append(combined_features)
                all_labels.append(label)
                all_study_ids.append(study_id)
                all_splits.append(split_name)

            except Exception as e:
                warnings.warn(f"Error processing {study_id}: {e}")
                continue

    # Build feature matrix
    print(f"\nBuilding feature matrix from {len(all_features)} subjects...")
    feature_names = list(all_features[0].keys())
    X = np.array([[f.get(name, np.nan) for name in feature_names] for f in all_features])
    y = np.array(all_labels)
    splits = np.array(all_splits)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features per image: {len(feature_names) // 2} (T1) + {len(feature_names) // 2} (T2)")

    # Handle NaN values
    nan_cols = np.any(np.isnan(X), axis=0)
    if np.any(nan_cols):
        print(f"Removing {np.sum(nan_cols)} features with NaN values")
        X = X[:, ~nan_cols]
        feature_names = [f for f, has_nan in zip(feature_names, nan_cols) if not has_nan]

    # Split data
    X_train = X[splits == 'train']
    y_train = y[splits == 'train']
    X_val = X[splits == 'val']
    y_val = y[splits == 'val']
    X_test = X[splits == 'test']
    y_test = y[splits == 'test']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Handle any remaining inf/nan after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)

    # Train classifier
    print("\nTraining Logistic Regression...")
    clf = LogisticRegressionCV(
        cv=5,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        scoring='roc_auc'
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    def evaluate(X, y, name):
        y_prob = clf.predict_proba(X)[:, 1]
        y_pred = clf.predict(X)

        auroc = roc_auc_score(y, y_prob)
        auprc = average_precision_score(y, y_prob)
        bal_acc = balanced_accuracy_score(y, y_pred)

        return auroc, auprc, bal_acc

    print("\n" + "=" * 50)
    print("=== Radiomic Features WMD Classification ===")
    print("=" * 50)
    print(f"Total features: {X.shape[1]}")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Best regularization C: {clf.C_[0]:.4f}")

    for name, X_split, y_split in [
        ('Train', X_train_scaled, y_train),
        ('Val', X_val_scaled, y_val),
        ('Test', X_test_scaled, y_test)
    ]:
        auroc, auprc, bal_acc = evaluate(X_split, y_split, name)
        print(f"\n{name} Results:")
        print(f"  AUROC: {auroc:.3f}")
        print(f"  AUPRC: {auprc:.3f}")
        print(f"  Balanced Accuracy: {bal_acc:.3f}")

    # Save features if requested
    if args.output_csv:
        features_df = pd.DataFrame(X, columns=feature_names)
        features_df['STUDY_ID'] = all_study_ids
        features_df['idWMD'] = all_labels
        features_df['split'] = all_splits
        features_df.to_csv(args.output_csv, index=False)
        print(f"\nFeatures saved to {args.output_csv}")


if __name__ == '__main__':
    main()
