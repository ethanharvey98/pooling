import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

# Add 3DINO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '3DINO'))

# Using HuggingFace download (auto-downloads and caches weights):
# python ../src/encode_oasis-3_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=1001' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --hf_download --seed=1001
# python ../src/encode_oasis-3_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=2001' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --hf_download --seed=2001
# python ../src/encode_oasis-3_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/3DINO_ViT/seed=3001' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --hf_download --seed=3001
#
# Using local weights:
# python ../src/encode_oasis-3_3dino.py --encoded_dir='/cluster/.../3DINO_ViT/seed=1001' --numpy_dir='/cluster/.../OASIS-3_MRI_numpy' --pretrained_weights='/path/to/3dino_vit_weights.pth' --seed=1001


def load_3dino_model(pretrained_weights):
    """Load 3DINO ViT-Large and return model on available device."""
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

    # Patch for Python <3.9 which lacks str.removesuffix
    if not hasattr(str, 'removesuffix'):
        def _removesuffix(self, suffix):
            if suffix and self.endswith(suffix):
                return self[:-len(suffix)]
            return self
        str.removesuffix = _removesuffix

    cfg = load_and_merge_config_3d('train/vit3d_highres')
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    return model


def normalize_volume(volume):
    """Percentile-based normalization to [-1, 1] (as in 3DINO notebook)."""
    min_val = torch.quantile(volume.float(), 0.0005)
    max_val = torch.quantile(volume.float(), 0.9995)
    volume = (volume - min_val) / (max_val - min_val + 1e-8)
    volume = torch.clip(volume * 2 - 1, -1, 1)
    return volume


def load_and_resample_volume(path, target_size=(112, 112, 112)):
    """Load .npz file as 3D volume and resample to target size.

    The existing .npz files contain arrays of shape (H, W, D) or (H, W, D, C).
    We take the first channel if multi-channel, resample to 112^3, and return
    shape (1, 1, 112, 112, 112).
    """
    data = np.load(path)
    arr = data['arr_0']  # (H, W, D) or (H, W, D, C)

    if arr.ndim == 4:
        # Multi-channel: take first channel (e.g., T1w)
        arr = arr[..., 0]

    volume = torch.as_tensor(arr, dtype=torch.float32)
    # (H, W, D) -> (1, 1, H, W, D) for interpolate
    volume = volume.unsqueeze(0).unsqueeze(0)
    volume = F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
    volume = normalize_volume(volume)
    return volume


def encode_split(model, df, device):
    """Encode all volumes in a dataframe split, return X, lengths, y."""
    X, lengths, y = [], [], []

    for i, row in df.iterrows():
        volume = load_and_resample_volume(row['path'])  # (1, 1, 112, 112, 112)
        volume = volume.to(device)

        with torch.no_grad():
            embedding = model(volume)  # (1, 1024)

        X.append(embedding.cpu())
        lengths.append(1)
        y.append(torch.tensor([row["Alzheimer's"]], dtype=torch.float32))

        if len(X) % 50 == 0:
            print(f"  Encoded {len(X)}/{len(df)} volumes")

    print(f"  Encoded {len(X)}/{len(df)} volumes (done)")
    return torch.cat(X), tuple(lengths), torch.stack(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode OASIS-3 volumes with 3DINO ViT.')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str, required=True)
    parser.add_argument('--numpy_dir', help='Directory to numpy dataset', type=str, required=True)
    parser.add_argument('--pretrained_weights', help='Path to 3DINO pretrained weights', type=str, default=None)
    parser.add_argument('--hf_download', action='store_true', default=False, help='Download weights from HuggingFace (AICONSlab/3DINO-ViT)')
    parser.add_argument('--seed', default=42, help='Random seed (default: 42)', type=int)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    # Resolve pretrained weights path
    if args.hf_download:
        from huggingface_hub import hf_hub_download
        args.pretrained_weights = hf_hub_download(
            repo_id="AICONSlab/3DINO-ViT",
            filename="3dino_vit_weights.pth",
        )
        print(f"Downloaded weights to: {args.pretrained_weights}")
    assert args.pretrained_weights is not None, "Provide --pretrained_weights or --hf_download"

    # --- Data splitting (same logic as encode_oasis-3.py) ---
    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')

    grouped_df = labels_df.groupby('Subject')["Alzheimer's"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Subject'], grouped_df["Alzheimer's"]
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(
        ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels
    )
    train_ids, val_ids = train_test_split(
        train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels
    )

    train_df = labels_df[labels_df['Subject'].isin(train_ids)]
    val_df = labels_df[labels_df['Subject'].isin(val_ids)]
    test_df = labels_df[labels_df['Subject'].isin(test_ids)]

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- Load 3DINO model ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_3dino_model(args.pretrained_weights)
    model.to(device)
    print(f"3DINO ViT-Large loaded. Embed dim: {model.embed_dim}")

    # --- Encode each split ---
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nEncoding {split_name} split ({len(split_df)} volumes)...")
        X, lengths, y = encode_split(model, split_df, device)
        save_path = f'{args.encoded_dir}/{split_name}.pth'
        torch.save({'X': X, 'lengths': lengths, 'y': y}, save_path)
        print(f"Saved {save_path}: X={X.shape}, y={y.shape}")
