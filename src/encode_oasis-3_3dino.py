import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import utils

# python ../src/encode_oasis-3_3dino.py --encoded_dir='...' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=1001 --hf_download --avgpool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode OASIS-3 volumes with 3DINO.')
    parser.add_argument('--encoded_dir', required=True, type=str)
    parser.add_argument('--numpy_dir', required=True, type=str)
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

    # --- Patient-level split ---
    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')

    grouped_df = labels_df.groupby('Subject')["Alzheimer's"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Subject'], grouped_df["Alzheimer's"]
    train_and_val_ids, test_ids, train_and_val_id_labels, _ = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)

    train_df = labels_df[labels_df['Subject'].isin(train_ids)]
    val_df = labels_df[labels_df['Subject'].isin(val_ids)]
    test_df = labels_df[labels_df['Subject'].isin(test_ids)]

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
            volumes = utils.load_and_resample_volume(row['path'])
            embedding = torch.cat([
                utils.encode_image_3dino(model, v, device, args.n_last_blocks, args.avgpool)
                for v in volumes
            ], dim=-1)

            X.append(embedding)
            lengths.append(1)
            y.append(torch.tensor([row["Alzheimer's"]], dtype=torch.float32))

            # Save axial slices for the first volume to verify orientation
            if not saved_slices:
                vol = volumes[0][0, 0]  # (H, W, D)
                n_slices = vol.shape[-1]
                indices = list(range(0, n_slices, max(1, n_slices // 12)))[:12]
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
