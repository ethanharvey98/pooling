import argparse
import os
import pandas as pd
import torch
import torchvision
# Importing our custom module(s)
import datasets
import utils

# python ../src/encode_kpsc.py --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5
# python ../src/encode_kpsc.py --encoder='3DINO' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/3DINO_ViT_concat/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5 --n_last_blocks=4 --avgpool --hf_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode_kpsc.py')
    parser.add_argument('--encoded_dir', required=True, type=str)
    parser.add_argument('--encoder', required=True, type=str, choices=['ViT-B/16', 'ConvNeXt-Tiny', 'MedSAM', '3DINO'])
    parser.add_argument('--numpy_dir', required=True, type=str)
    parser.add_argument('--test_site_ids', required=True, nargs='+', type=int)
    parser.add_argument('--train_site_ids', required=True, nargs='+', type=int)
    parser.add_argument('--val_site_ids', required=True, nargs='+', type=int)
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool', action='store_true', default=False)
    parser.add_argument('--pretrained_weights', default=None, type=str)
    parser.add_argument('--hf_download', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    # --- Data splitting (site-level) ---
    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')
    label_cols = ['idCBI', 'idWMD']

    train_df = labels_df[labels_df['SiteID'].isin(args.train_site_ids)]
    val_df = labels_df[labels_df['SiteID'].isin(args.val_site_ids)]
    test_df = labels_df[labels_df['SiteID'].isin(args.test_site_ids)]

    # --- Model ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.encoder == 'ViT-B/16':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        model.conv_proj.weight.data = model.conv_proj.weight.data.sum(dim=1, keepdim=True)
        model.conv_proj.in_channels = 1
        model.heads = torch.nn.Identity()
    elif args.encoder == 'ConvNeXt-Tiny':
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1, keepdim=True)
        model.features[0][0].in_channels = 1
        model.classifier[2] = torch.nn.Identity()
    elif args.encoder == 'MedSAM':
        from segment_anything import sam_model_registry
        checkpoint = torch.load('/cluster/tufts/hugheslab/eharve06/pooling/models/medsam_vit_b.pth', map_location='cpu', weights_only=False)
        medsam = sam_model_registry['vit_b']()
        medsam.load_state_dict(checkpoint)
        model = medsam.image_encoder
        model.patch_embed.proj.weight.data = model.patch_embed.proj.weight.data.sum(dim=1, keepdim=True)
        model.patch_embed.proj.in_channels = 1
        model.neck.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    elif args.encoder == '3DINO':
        if args.hf_download:
            from huggingface_hub import hf_hub_download
            args.pretrained_weights = hf_hub_download(repo_id="AICONSlab/3DINO-ViT", filename="3dino_vit_weights.pth")
        model = utils.load_3dino_model(args.pretrained_weights)

    model.to(device)

    # --- Build encode function ---
    if args.encoder == '3DINO':
        def encode_fn(path):
            volumes = utils.load_and_resample_volume(path)
            embedding = torch.cat([utils.encode_image_3dino(model, v, device, args.n_last_blocks, args.avgpool) for v in volumes], dim=-1)
            return embedding, 1
    else:
        resize_size = 1024 if args.encoder == 'MedSAM' else 224
        transform = torchvision.transforms.Compose([
            lambda path: utils.read_npz(path),
            lambda image: image.permute(3, 0, 1, 2),
            lambda image: utils.pad_image(image),
            torchvision.transforms.Resize(size=(resize_size, resize_size)),
            lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
        ])

        # Compute mean/std from training set
        train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[label_cols].values, dtype=torch.float32), transform)
        means, stds = [], []
        for image, _, _ in train_dataset:
            means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
            stds.append(torch.std(image, dim=(0, 2, 3)).tolist())
        mean = torch.tensor(means).mean(dim=0)
        std = torch.tensor(stds).mean(dim=0)

        transform = torchvision.transforms.Compose([
            lambda path: utils.read_npz(path),
            lambda image: image.permute(3, 0, 1, 2),
            lambda image: utils.pad_image(image),
            lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
            torchvision.transforms.Resize(size=(resize_size, resize_size)),
            lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
        ])

        def encode_fn(path):
            image = transform(path)
            embeddings = torch.cat([utils.encode_image(model, image[:, c].unsqueeze(1)) for c in range(image.shape[1])], dim=-1)
            return embeddings, len(image)

    # --- Encode all splits ---
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        x, lengths, y = [], [], []
        for _, row in split_df.iterrows():
            embedding, length = encode_fn(row['path'])
            x.append(embedding)
            lengths.append(length)
            y.append(torch.tensor([row[col] for col in label_cols], dtype=torch.float32))
        torch.save({'X': torch.cat(x), 'lengths': tuple(lengths), 'y': torch.stack(y)}, f'{args.encoded_dir}/{split_name}.pth')
        print(f"{split_name}: {torch.cat(x).shape}")
