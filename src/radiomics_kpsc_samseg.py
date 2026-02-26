#!/usr/bin/env python3
"""
Radiomic feature extraction and WMD classification for KPSC MRI data.
Uses raw NIfTI files with SAMSEG segmentation masks for white matter ROI.

Usage:
    python radiomics_kpsc_samseg.py \
        --nifti_dir /path/to/KPSC_MRI_800_nifti \
        --samseg_t1_dir /path/to/SAMSEG_KPSC_MRI_800 \
        --samseg_t2_dir /path/to/SAMSEG_KPSC_MRI_800_T2
"""

import argparse
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from tqdm import tqdm
import warnings

# Suppress radiomics logging
import logging
logging.getLogger('radiomics').setLevel(logging.ERROR)

# SAMSEG white matter labels
WM_LABELS = [
    2,   # Left Cerebral White Matter
    41,  # Right Cerebral White Matter
    7,   # Left Cerebellum White Matter
    46,  # Right Cerebellum White Matter
    77,  # WM Hypointensities (lesions)
]


def load_nifti_as_sitk(nifti_path):
    """Load NIfTI file and convert to SimpleITK image."""
    return sitk.ReadImage(nifti_path, sitk.sitkFloat32)


def load_samseg_wm_mask(seg_path, wm_labels=WM_LABELS):
    """Load SAMSEG seg.mgz and create binary WM+lesion mask."""
    # Load with nibabel (handles .mgz format)
    seg_nib = nib.load(seg_path)
    seg_data = seg_nib.get_fdata().astype(np.int32)

    # Create binary mask from WM labels
    mask_data = np.zeros_like(seg_data, dtype=np.uint8)
    for label in wm_labels:
        mask_data[seg_data == label] = 1

    # Convert to SimpleITK
    # nibabel uses (x, y, z) ordering, need to transpose for SimpleITK (z, y, x)
    mask_transposed = np.transpose(mask_data, (2, 1, 0))
    mask_sitk = sitk.GetImageFromArray(mask_transposed)

    # Copy spatial information from the original segmentation
    seg_sitk = sitk.ReadImage(seg_path)
    mask_sitk.CopyInformation(seg_sitk)

    return mask_sitk


def extract_features(image_sitk, mask_sitk, extractor, prefix=''):
    """Extract radiomic features from image with mask."""
    try:
        # Resample mask to image space if needed
        if image_sitk.GetSize() != mask_sitk.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image_sitk)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_sitk = resampler.Execute(mask_sitk)

        # Ensure mask has correct type
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

        # Check if mask has any foreground voxels
        mask_array = sitk.GetArrayFromImage(mask_sitk)
        if np.sum(mask_array) == 0:
            warnings.warn(f"Empty mask for {prefix}, skipping")
            return None

        features = extractor.execute(image_sitk, mask_sitk)

        # Filter to only feature values (not diagnostics)
        feature_dict = {}
        for key, val in features.items():
            if not key.startswith('diagnostics'):
                feature_name = f"{prefix}_{key}" if prefix else key
                feature_dict[feature_name] = float(val)
        return feature_dict

    except Exception as e:
        warnings.warn(f"Feature extraction failed for {prefix}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Radiomic features for WMD classification (SAMSEG mask)'
    )
    parser.add_argument(
        '--nifti_dir',
        default='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti',
        help='Path to KPSC_MRI_800_nifti directory'
    )
    parser.add_argument(
        '--samseg_t1_dir',
        default='/cluster/tufts/hugheslabkp/data_irb_required/SAMSEG_KPSC_MRI_800',
        help='Path to SAMSEG T1 outputs'
    )
    parser.add_argument(
        '--samseg_t2_dir',
        default='/cluster/tufts/hugheslabkp/data_irb_required/SAMSEG_KPSC_MRI_800_T2',
        help='Path to SAMSEG T2 outputs'
    )
    parser.add_argument(
        '--labels_csv',
        default='/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800/ADDF_2024_External_MRimage_800.csv',
        help='Path to labels CSV'
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
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Loaded {len(labels_df)} subjects from labels CSV")

    # Site-based splitting
    train_df = labels_df[labels_df['SiteID'].isin(args.train_site_ids)]
    val_df = labels_df[labels_df['SiteID'].isin(args.val_site_ids)]
    test_df = labels_df[labels_df['SiteID'].isin(args.test_site_ids)]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Initialize PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
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

            # Build paths
            t1_path = os.path.join(args.nifti_dir, study_id, 'T1.nii.gz')
            t2_path = os.path.join(args.nifti_dir, study_id, 'T2.nii.gz')
            samseg_t1_path = os.path.join(args.samseg_t1_dir, study_id, 'seg.mgz')
            samseg_t2_path = os.path.join(args.samseg_t2_dir, study_id, 'seg.mgz')

            # Check if all files exist
            if not all(os.path.exists(p) for p in [t1_path, t2_path, samseg_t1_path, samseg_t2_path]):
                missing = [p for p in [t1_path, t2_path, samseg_t1_path, samseg_t2_path] if not os.path.exists(p)]
                warnings.warn(f"Skipping {study_id}, missing files: {missing}")
                continue

            try:
                # Load images
                t1_sitk = load_nifti_as_sitk(t1_path)
                t2_sitk = load_nifti_as_sitk(t2_path)

                # Load SAMSEG masks
                t1_mask = load_samseg_wm_mask(samseg_t1_path)
                t2_mask = load_samseg_wm_mask(samseg_t2_path)

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

    if len(all_features) == 0:
        print("No features extracted! Check your data paths.")
        return

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
    print("=== Radiomic Features WMD Classification (SAMSEG) ===")
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
