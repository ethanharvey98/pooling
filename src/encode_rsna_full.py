import argparse
import ast
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
import datasets
import utils

# python ../src/encode_rsna_full.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA/ViT_B_16/seed=1001' --encoder='ViT-B/16' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --seed=1001

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode_rsna_full.py')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str)
    parser.add_argument('--encoder', help='Encoder pre-trained on ImageNet', type=str)
    parser.add_argument('--labels_csv', help='Path to labels.csv', type=str)
    parser.add_argument('--numpy_dir', help='Directory to numpy dataset', type=str)
    parser.add_argument('--seed', default=42, help='Random seed (default: 42)', type=int)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    # Load labels.csv
    labels_df = pd.read_csv(args.labels_csv)

    # Get scan-level labels (1 if any slice has hemorrhage)
    labels_df['scan_label'] = labels_df['Any'].apply(lambda x: 1 if any(ast.literal_eval(x)) else 0)
    labels_df['path'] = labels_df['Study ID'].apply(lambda x: f'{args.numpy_dir}/{x}.npz')

    # Filter to only scans with existing npz files
    labels_df = labels_df[labels_df['path'].apply(os.path.exists)]
    print(f"Scans with npz files: {len(labels_df)}")

    # Patient-level train/val/test split (4/6 train, 1/6 val, 1/6 test)
    grouped_df = labels_df.groupby('Patient ID')['scan_label'].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Patient ID'], grouped_df['scan_label']
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)

    train_df = labels_df[labels_df['Patient ID'].isin(train_ids)]
    val_df = labels_df[labels_df['Patient ID'].isin(val_ids)]
    test_df = labels_df[labels_df['Patient ID'].isin(test_ids)]

    print(f"Train: {len(train_df)} ({train_df['scan_label'].sum()} positive)")
    print(f"Val: {len(val_df)} ({val_df['scan_label'].sum()} positive)")
    print(f"Test: {len(test_df)} ({test_df['scan_label'].sum()} positive)")

    resize_size = 1024 if args.encoder == 'MedSAM' else 224
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.permute(3, 0, 1, 2),  # (C, H, W, S) -> (S, C, H, W)
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(resize_size, resize_size)),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['scan_label']].values, dtype=torch.float32), transform)

    means, stds = [], []

    for image, num_slices, label in train_dataset:
        means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
        stds.append(torch.std(image, dim=(0, 2, 3)).tolist())

    mean = torch.tensor(means).mean(dim=0)
    std = torch.tensor(stds).mean(dim=0)

    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.permute(3, 0, 1, 2),  # (C, H, W, S) -> (S, C, H, W)
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(resize_size, resize_size)),
        lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['scan_label']].values, dtype=torch.float32), transform)
    val_dataset = datasets.MILPathDataset(val_df.path.values, torch.tensor(val_df[['scan_label']].values, dtype=torch.float32), transform)
    test_dataset = datasets.MILPathDataset(test_df.path.values, torch.tensor(test_df[['scan_label']].values, dtype=torch.float32), transform)

    assert args.encoder in ['ViT-B/16', 'ConvNeXt-Tiny', 'MedSAM']
    if args.encoder == 'ViT-B/16':
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
        model.conv_proj.weight.data = model.conv_proj.weight.data.sum(dim=1, keepdim=True)
        model.conv_proj.in_channels = 1
        model.heads = torch.nn.Identity()
    elif args.encoder == 'ConvNeXt-Tiny':
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights(weights))
        model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1, keepdim=True)
        model.features[0][0].in_channels = 1
        model.classifier[2] = torch.nn.Identity()
    elif args.encoder == 'MedSAM':
        from segment_anything import sam_model_registry
        checkpoint_path = '/cluster/tufts/hugheslab/eharve06/pooling/models/medsam_vit_b.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        medsam = sam_model_registry['vit_b']()
        medsam.load_state_dict(checkpoint)
        model = medsam.image_encoder
        model.patch_embed.proj.weight.data = model.patch_embed.proj.weight.data.sum(dim=1, keepdim=True)
        model.patch_embed.proj.in_channels = 1
        model.neck.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    for split_name, split_df, split_dataset in [
        ('train', train_df, train_dataset),
        ('val', val_df, val_dataset),
        ('test', test_df, test_dataset),
    ]:
        X, lengths, y, instance_y = [], [], [], []
        any_lists = [ast.literal_eval(row['Any']) for _, row in split_df.iterrows()]

        for i, (image, length, label) in enumerate(split_dataset):
            embeddings = torch.cat([
                utils.encode_image(model, image[:,c].unsqueeze(1))
                for c in range(image.shape[1])
            ], dim=-1)
            X.append(embeddings)
            lengths.append(length)
            y.append(label)
            if len(any_lists[i]) == length:
                instance_y.append(torch.tensor(any_lists[i], dtype=torch.float32))
            else:
                print(f"WARNING: {split_name} scan {i} has {length} slices but {len(any_lists[i])} labels, skipping instance labels")
                instance_y.append(torch.full((length,), -1.0))

        torch.save({
            'X': torch.cat(X),
            'lengths': tuple(lengths),
            'y': torch.stack(y),
            'instance_y': torch.cat(instance_y),
        }, f'{args.encoded_dir}/{split_name}.pth')
