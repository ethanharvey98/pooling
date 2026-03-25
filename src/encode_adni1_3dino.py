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

# 4-block concat + avgpool (as recommended by 3DINO paper):
# python ../src/encode_adni1_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_ADNI1_Complete_1Yr_1.5T/3DINO_ViT_concat/seed=1001' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --hf_download --seed=1001 --n_last_blocks=4 --avgpool
# python ../src/encode_adni1_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_ADNI1_Complete_1Yr_1.5T/3DINO_ViT_concat/seed=2001' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --hf_download --seed=2001 --n_last_blocks=4 --avgpool
# python ../src/encode_adni1_3dino.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_ADNI1_Complete_1Yr_1.5T/3DINO_ViT_concat/seed=3001' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --hf_download --seed=3001 --n_last_blocks=4 --avgpool


def load_3dino_model(pretrained_weights):
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

    cfg = load_and_merge_config_3d('train/vit3d_highres')
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    return model


def normalize_volume(volume):
    min_val = torch.quantile(volume.float(), 0.0005)
    max_val = torch.quantile(volume.float(), 0.9995)
    volume = (volume - min_val) / (max_val - min_val + 1e-8)
    volume = torch.clip(volume * 2 - 1, -1, 1)
    return volume


def load_and_resample_volume(path, target_size=(112, 112, 112)):
    data = np.load(path)
    arr = data['arr_0']

    if arr.ndim == 4:
        channels = [arr[c] for c in range(arr.shape[0])]
    else:
        channels = [arr]

    volumes = []
    for ch in channels:
        volume = torch.as_tensor(ch, dtype=torch.float32)
        volume = volume.unsqueeze(0).unsqueeze(0)
        volume = F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
        volume = normalize_volume(volume)
        volumes.append(volume)
    return volumes


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (output, torch.mean(intermediate_output[-1][0], dim=1)),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


def encode_volume(model, volume, device, n_last_blocks, avgpool):
    volume = volume.to(device)
    with torch.no_grad():
        if n_last_blocks > 1 or avgpool:
            features = model.get_intermediate_layers(
                volume, n_last_blocks, return_class_token=True
            )
            return create_linear_input(features, n_last_blocks, avgpool)
        else:
            return model(volume)


def encode_split(model, df, device, n_last_blocks, avgpool):
    X, lengths, y = [], [], []

    for i, row in df.iterrows():
        volumes = load_and_resample_volume(row['path'])

        channel_embeddings = []
        for volume in volumes:
            emb = encode_volume(model, volume, device, n_last_blocks, avgpool)
            channel_embeddings.append(emb)

        embedding = torch.cat(channel_embeddings, dim=-1)

        X.append(embedding.cpu())
        lengths.append(1)
        y.append(torch.tensor([row["Alzheimer's"]], dtype=torch.float32))

        if len(X) % 50 == 0:
            print(f"  Encoded {len(X)}/{len(df)} volumes")

    print(f"  Encoded {len(X)}/{len(df)} volumes (done)")
    return torch.cat(X), tuple(lengths), torch.stack(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode ADNI1 volumes with 3DINO ViT.')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str, required=True)
    parser.add_argument('--numpy_dir', help='Directory to numpy dataset', type=str, required=True)
    parser.add_argument('--pretrained_weights', help='Path to 3DINO pretrained weights', type=str, default=None)
    parser.add_argument('--hf_download', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_last_blocks', default=1, type=int)
    parser.add_argument('--avgpool', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    if args.hf_download:
        from huggingface_hub import hf_hub_download
        args.pretrained_weights = hf_hub_download(
            repo_id="AICONSlab/3DINO-ViT",
            filename="3dino_vit_weights.pth",
        )
        print(f"Downloaded weights to: {args.pretrained_weights}")
    assert args.pretrained_weights is not None, "Provide --pretrained_weights or --hf_download"

    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')

    grouped_df = labels_df.groupby('Subject')["Alzheimer's"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Subject'], grouped_df["Alzheimer's"]
    train_and_val_ids, test_ids, train_and_val_id_labels, _ = train_test_split(
        ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels
    )
    train_ids, val_ids = train_test_split(
        train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels
    )

    train_df = labels_df[labels_df['Subject'].isin(train_ids)]
    val_df = labels_df[labels_df['Subject'].isin(val_ids)]
    test_df = labels_df[labels_df['Subject'].isin(test_ids)]

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_3dino_model(args.pretrained_weights)
    model.to(device)

    embed_dim = model.embed_dim
    per_channel_dim = embed_dim * args.n_last_blocks + (embed_dim if args.avgpool else 0)
    print(f"3DINO ViT-Large loaded. n_last_blocks={args.n_last_blocks}, avgpool={args.avgpool}, per_channel_dim={per_channel_dim}")

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nEncoding {split_name} split ({len(split_df)} volumes)...")
        X, lengths, y = encode_split(model, split_df, device, args.n_last_blocks, args.avgpool)
        save_path = f'{args.encoded_dir}/{split_name}.pth'
        torch.save({'X': X, 'lengths': lengths, 'y': y}, save_path)
        print(f"Saved {save_path}: X={X.shape}, y={y.shape}")
