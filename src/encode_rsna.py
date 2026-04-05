import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
# Importing our custom module(s)
import datasets
import utils

# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/ViT_B_16/seed=1001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/dloevl01/datasets/RSNA/RSNA_ICH_numpy' --csv_path='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' --seed=1001
# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/ViT_B_16/seed=2001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/dloevl01/datasets/RSNA/RSNA_ICH_numpy' --csv_path='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' --seed=2001
# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/dloevl01/encoded_RSNA_ICH/ViT_B_16/seed=3001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/dloevl01/datasets/RSNA/RSNA_ICH_numpy' --csv_path='/home/denny-loevlie/RSNA_Investigation/rsna_ich_subset_1149_complete.csv' --seed=3001

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode_rsna.py')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str)
    parser.add_argument('--encoder', help='Encoder pre-trained on ImageNet', type=str)
    parser.add_argument('--numpy_dir', help='Directory to numpy dataset', type=str)
    parser.add_argument('--csv_path', help='Path to CSV with scan labels', type=str)
    parser.add_argument('--labels_csv', help='Path to full RSNA labels.csv (for Patient ID)', type=str)
    parser.add_argument('--seed', default=42, help='Random seed (default: 42)', type=int)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    # Load CSV and get scan-level labels
    df = pd.read_csv(args.csv_path)
    labels_df = df.groupby('scan_id').agg({'scan_label': 'first'}).reset_index()
    labels_df['path'] = labels_df['scan_id'].apply(lambda x: f'{args.numpy_dir}/{x}.npz')

    # Get Patient ID from full labels.csv
    full_labels_df = pd.read_csv(args.labels_csv)
    patient_map = dict(zip(full_labels_df['Study ID'], full_labels_df['Patient ID']))
    labels_df['Patient ID'] = labels_df['scan_id'].map(patient_map)

    # Build instance label map (scan_id -> sorted slice labels)
    instance_label_map = {
        scan_id: group.sort_values('slice_order')['label_any'].values
        for scan_id, group in df.groupby('scan_id')
    }

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

    # Transform: RSNA is (num_slices, H, W), need to add channel dim
    # OASIS-3 permutes (3, 0, 1, 2) because it has shape (D, H, W, C)
    # RSNA has shape (num_slices, H, W) so we just add a channel dim
    resize_size = 1024 if args.encoder == 'MedSAM' else 224
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.unsqueeze(1),  # (D, H, W) -> (D, 1, H, W)
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
        lambda image: image.unsqueeze(1),  # (D, H, W) -> (D, 1, H, W)
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
        scan_ids = split_df['scan_id'].values

        for i, (image, length, label) in enumerate(split_dataset):
            embeddings = torch.cat([
                utils.encode_image(model, image[:,c].unsqueeze(1))
                for c in range(image.shape[1])
            ], dim=-1)
            X.append(embeddings)
            lengths.append(length)
            y.append(label)
            instance_y.append(torch.tensor(instance_label_map[scan_ids[i]][:length], dtype=torch.float32))

        torch.save({
            'X': torch.cat(X),
            'lengths': tuple(lengths),
            'y': torch.stack(y),
            'instance_y': torch.cat(instance_y),
        }, f'{args.encoded_dir}/{split_name}.pth')
