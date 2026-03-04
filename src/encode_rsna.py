import argparse
import ast
import os

os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.hf_cache')
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
# Importing our custom module(s)
import datasets
import utils

# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/ViT_B_16/seed=1001' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001
# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/Qwen2.5-VL/seed=1001' --encoder='Qwen2.5-VL' --model_name='Qwen/Qwen2.5-VL-7B-Instruct' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001
# python ../src/encode_rsna.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_RSNA/QoQ-Med-VL/seed=1001' --encoder='Qwen2.5-VL' --model_name='ddvd233/QoQ-Med-VL-7B' --numpy_dir='/cluster/tufts/hugheslab/datasets/RSNA_numpy' --labels_csv='/cluster/tufts/hugheslab/datasets/RSNA/labels.csv' --seed=1001


def encode_qwen_slices(vision_encoder, slices, device):
    """Encode a batch of grayscale slices using Qwen2.5-VL vision encoder.

    Args:
        vision_encoder: The Qwen2.5-VL visual module.
        slices: Tensor of shape (num_slices, 1, H, W) — single-channel grayscale.
        device: Torch device.

    Returns:
        Tensor of shape (num_slices, 3584) — one embedding per slice.
    """
    vision_encoder.eval()
    num_slices = slices.shape[0]
    H, W = slices.shape[2], slices.shape[3]
    patch_size = 14
    temporal_patch_size = 2

    grid_h = H // patch_size
    grid_w = W // patch_size

    embeddings = []
    with torch.no_grad():
        # Process slices in pairs (temporal_patch_size=2 requires temporal dim >= 2)
        for i in range(0, num_slices, 2):
            if i + 1 < num_slices:
                pair = slices[i:i+2]  # (2, 1, H, W)
            else:
                # Odd last slice: duplicate it
                pair = slices[i:i+1].repeat(2, 1, 1, 1)  # (2, 1, H, W)

            # Replicate grayscale to 3 channels: (2, 3, H, W)
            pair_rgb = pair.repeat(1, 3, 1, 1)

            # Reshape to (3, 2, H, W) for Conv3d: (channel, temporal, height, width)
            pixel_values = pair_rgb.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 2, H, W)
            # Flatten to (num_patches_temporal * num_patches_h * num_patches_w, 3 * temporal_patch_size * patch_size * patch_size)
            # But the Qwen visual encoder expects pixel_values as (seq_len, patch_dim)
            # Actually, let's just pass the raw pixels and grid_thw

            # grid_thw: (num_videos, temporal_tokens, height_tokens, width_tokens)
            t_tokens = 2 // temporal_patch_size  # = 1
            grid_thw = torch.tensor([[t_tokens, grid_h, grid_w]], dtype=torch.long, device=device)

            # pixel_values needs to be (total_patches, channel_dim)
            # channel_dim = 3 * temporal_patch_size * patch_size * patch_size
            # Let's reshape pixel_values properly
            # From (2, 3, H, W) -> extract patches
            channel_dim = 3 * temporal_patch_size * patch_size * patch_size

            # Reshape: (temporal, channels, H, W) -> patches
            # (2, 3, H, W) -> (1, 3, 2, H, W) -> unfold
            pixels = pair_rgb.to(device, dtype=torch.float16)  # (2, 3, H, W)
            # Rearrange to (3, 2, grid_h, patch_size, grid_w, patch_size)
            pixels = pixels.permute(1, 0, 2, 3)  # (3, 2, H, W)
            pixels = pixels.reshape(3, temporal_patch_size, grid_h, patch_size, grid_w, patch_size)
            pixels = pixels.permute(2, 4, 1, 0, 3, 5)  # (grid_h, grid_w, t_patch, 3, patch_h, patch_w)
            pixels = pixels.reshape(-1, channel_dim)  # (grid_h * grid_w, channel_dim)

            output = vision_encoder(pixels, grid_thw=grid_thw)
            # output shape: (num_tokens, hidden_dim=3584)
            # Mean-pool all tokens to get one embedding per temporal group
            embedding = output.mean(dim=0).float().cpu()  # (3584,)

            if i + 1 < num_slices:
                # Two real slices — assign same embedding to both
                embeddings.append(embedding)
                embeddings.append(embedding)
            else:
                # Only one real slice
                embeddings.append(embedding)

    return torch.stack(embeddings)  # (num_slices, 3584)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode_rsna.py')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str)
    parser.add_argument('--encoder', help='Encoder type', type=str)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-VL-7B-Instruct', help='HuggingFace model name for Qwen2.5-VL (default: Qwen/Qwen2.5-VL-7B-Instruct)', type=str)
    parser.add_argument('--numpy_dir', help='Directory to RSNA numpy dataset', type=str)
    parser.add_argument('--labels_csv', help='Path to RSNA labels.csv', type=str)
    parser.add_argument('--seed', default=42, help='Random seed (default: 42)', type=int)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    labels_df = pd.read_csv(args.labels_csv)

    # Parse the 'Any' column (list of per-slice binary labels) and compute bag-level label
    labels_df['Any_list'] = labels_df['Any'].apply(ast.literal_eval)
    labels_df['bag_label'] = labels_df['Any_list'].apply(max)

    # Build paths to numpy files
    labels_df['path'] = labels_df['Study'].apply(lambda sid: f'{args.numpy_dir}/{sid}.npz')

    # Split by Study ID
    ids = labels_df['Study']
    id_labels = labels_df['bag_label']
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)

    train_df = labels_df[labels_df['Study'].isin(train_ids)]
    val_df = labels_df[labels_df['Study'].isin(val_ids)]
    test_df = labels_df[labels_df['Study'].isin(test_ids)]

    # Determine resize target based on encoder
    if args.encoder == 'Qwen2.5-VL':
        resize_size = 448
    else:
        resize_size = 224

    # RSNA numpy format: (num_slices, 512, 512) — no channel dim
    # Transform: read -> add channel dim -> pad -> resize -> normalize
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.unsqueeze(1),  # (num_slices, 1, 512, 512)
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(resize_size, resize_size)),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['bag_label']].values, dtype=torch.float32), transform)

    # Compute normalization stats from training set
    means, stds = [], []
    for image, num_slices, label in train_dataset:
        means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
        stds.append(torch.std(image, dim=(0, 2, 3)).tolist())

    mean = torch.tensor(means).mean(dim=0)
    std = torch.tensor(stds).mean(dim=0)

    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.unsqueeze(1),  # (num_slices, 1, 512, 512)
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(resize_size, resize_size)),
        lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['bag_label']].values, dtype=torch.float32), transform)
    val_dataset = datasets.MILPathDataset(val_df.path.values, torch.tensor(val_df[['bag_label']].values, dtype=torch.float32), transform)
    test_dataset = datasets.MILPathDataset(test_df.path.values, torch.tensor(test_df[['bag_label']].values, dtype=torch.float32), transform)

    # Set up encoder
    assert args.encoder in ['ViT-B/16', 'ConvNeXt-Tiny', 'MedSAM', 'Qwen2.5-VL']
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
    elif args.encoder == 'Qwen2.5-VL':
        from transformers import Qwen2_5_VLForConditionalGeneration
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="cpu"
        )
        model = full_model.visual
        del full_model
        for param in model.parameters():
            param.requires_grad = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # Encode each split
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        X, lengths, y = [], [], []

        for image, length, label in dataset:
            if args.encoder == 'Qwen2.5-VL':
                embeddings = encode_qwen_slices(model, image, device)
            else:
                embeddings = torch.cat([
                    utils.encode_image(model, image[:,c].unsqueeze(1))
                    for c in range(image.shape[1])
                ], dim=-1)

            X.append(embeddings)
            lengths.append(length)
            y.append(label)

        torch.save({
            'X': torch.cat(X),
            'lengths': tuple(lengths),
            'y': torch.stack(y),
        }, f'{args.encoded_dir}/{split_name}.pth')

        print(f'Saved {split_name}.pth: X={torch.cat(X).shape}, lengths={len(lengths)}, y={torch.stack(y).shape}')
