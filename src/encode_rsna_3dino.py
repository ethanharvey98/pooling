import argparse
import ast
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils

# python ../src/encode_rsna_3dino.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/3DINO_ViT_concat/seed=1001' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy' --csv_path='/cluster/tufts/hugheslab/datasets/RSNA_ICH/subset_labels.csv' --pretrained_weights='...' --seed=1001 --avgpool


def load_and_resample_rsna_volume(path, target_size=(112, 112, 112)):
    """Load RSNA .npz (D, H, W) and resample to target size for 3DINO.

    RSNA CT volumes are stored as (num_slices, H, W) from preprocess_rsna.py.
    We transpose to (H, W, D) before resampling to match 3DINO's expected orientation.
    """
    data = np.load(path)
    arr = data['arr_0']  # (D, H, W)
    arr = np.transpose(arr, (1, 2, 0))  # (H, W, D)

    volume = torch.as_tensor(arr, dtype=torch.float32)
    volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    volume = F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
    volume = utils.normalize_volume_3dino(volume)
    return [volume]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode RSNA ICH 1149-subset volumes with 3DINO.')
    parser.add_argument('--encoded_dir', required=True, type=str)
    parser.add_argument('--numpy_dir', required=True, type=str)
    parser.add_argument('--csv_path', required=True, type=str,
                        help='Path to subset_labels.csv (Patient ID, Study ID, Slice ID, z, Any)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool', action='store_true', default=False)
    parser.add_argument('--pretrained_weights', default=None, type=str)
    parser.add_argument('--hf_download', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)
    os.makedirs(f'{args.encoded_dir}/slices', exist_ok=True)

    # --- Resolve weights ---
    if args.hf_download:
        from huggingface_hub import hf_hub_download
        args.pretrained_weights = hf_hub_download(repo_id="AICONSlab/3DINO-ViT", filename="3dino_vit_weights.pth")
    assert args.pretrained_weights is not None, "Provide --pretrained_weights or --hf_download"

    # --- Load and prepare labels (subset_labels.csv: Patient ID, Study ID, Slice ID, z, Any) ---
    labels_df = pd.read_csv(args.csv_path)
    columns = ['Slice ID', 'z', 'Any']
    labels_df[columns] = labels_df[columns].apply(lambda col: col.map(ast.literal_eval))
    labels_df['path'] = labels_df['Study ID'].apply(lambda study_id: f'{args.numpy_dir}/{study_id}.npz')
    labels_df['ICH'] = labels_df.apply(lambda row: float(any(row.Any)), axis=1)

    # --- Patient-level stratified split ---
    grouped_df = labels_df.groupby('Patient ID')['ICH'].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Patient ID'], grouped_df['ICH']
    train_and_val_ids, test_ids, train_and_val_id_labels, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels
    )
    train_ids, val_ids = train_test_split(
        train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels
    )

    train_df = labels_df[labels_df['Patient ID'].isin(train_ids)]
    val_df = labels_df[labels_df['Patient ID'].isin(val_ids)]
    test_df = labels_df[labels_df['Patient ID'].isin(test_ids)]

    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- Load model ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = utils.load_3dino_model(args.pretrained_weights)
    model.to(device)

    per_channel_dim = model.embed_dim * args.n_last_blocks + (model.embed_dim if args.avgpool else 0)
    print(f"n_last_blocks={args.n_last_blocks}, avgpool={args.avgpool}, per_channel_dim={per_channel_dim}")

    # --- Encode ---
    saved_slices = False
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        X, lengths, y = [], [], []

        for _, row in split_df.iterrows():
            volumes = load_and_resample_rsna_volume(row['path'])
            embedding = torch.cat([
                utils.encode_image_3dino(model, v, device, args.n_last_blocks, args.avgpool)
                for v in volumes
            ], dim=-1)

            X.append(embedding)
            lengths.append(1)
            y.append(torch.tensor([row['ICH']], dtype=torch.float32))

            # Save axial slices for the first volume to verify orientation
            if not saved_slices:
                vol = volumes[0][0, 0]  # (H, W, D)
                n_slices = vol.shape[-1]
                indices = [int(i) for i in torch.linspace(0, n_slices - 1, 12)]
                fig, axes = plt.subplots(1, len(indices), figsize=(2 * len(indices), 2))
                for i, idx in enumerate(indices):
                    axes[i].imshow(vol[:, :, idx].numpy(), cmap='gray')
                    axes[i].set_title(f'D={idx}', fontsize=8)
                    axes[i].axis('off')
                fig.suptitle('Axial slices fed to 3DINO (D=0 inferior -> superior)', fontsize=10)
                fig.tight_layout()
                fig.savefig(f'{args.encoded_dir}/slices/axial_orientation.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved orientation check: {args.encoded_dir}/slices/axial_orientation.png")
                saved_slices = True

            if len(X) % 50 == 0:
                print(f"  {split_name}: {len(X)}/{len(split_df)}")

        torch.save({'X': torch.cat(X), 'lengths': tuple(lengths), 'y': torch.stack(y)}, f'{args.encoded_dir}/{split_name}.pth')
        print(f"{split_name}: {torch.cat(X).shape}")
