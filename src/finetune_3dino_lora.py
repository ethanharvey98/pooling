import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add 3DINO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '3DINO'))


def load_3dino_model(pretrained_weights):
    """Load 3DINO ViT-Large and return model on available device."""
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

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
    """Load .npz file and return a list of volumes, one per channel."""
    data = np.load(path)
    arr = data['arr_0']  # (C, H, W, D) or (H, W, D)

    if arr.ndim == 4:
        channels = [arr[c] for c in range(arr.shape[0])]
    else:
        channels = [arr]

    volumes = []
    for ch in channels:
        volume = torch.as_tensor(ch, dtype=torch.float32)
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        volume = F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
        volume = normalize_volume(volume)
        volumes.append(volume)
    return volumes


def compute_metrics(all_logits, all_labels, criterion_fn, trainable_params, alpha):
    """Compute loss, nll, auroc, auprc, balanced accuracy."""
    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs = torch.sigmoid(logits_t).detach().cpu().numpy()
    labels_np = labels_t.cpu().numpy()

    nll = F.binary_cross_entropy_with_logits(logits_t, labels_t).item()

    # Regularization penalty (matches losses.py convention)
    if criterion_fn == 'L1':
        penalty = (alpha / 2) * torch.abs(trainable_params).sum().item()
    elif criterion_fn == 'L2':
        penalty = (alpha / 2) * (trainable_params ** 2).sum().item()
    else:
        penalty = 0.0

    loss = nll + penalty

    try:
        auroc = roc_auc_score(labels_np, probs)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(labels_np, probs)
    except ValueError:
        auprc = 0.0

    preds = (probs >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(labels_np, preds)

    return {'loss': loss, 'nll': nll, 'auroc': auroc, 'auprc': auprc, 'bal_acc': bal_acc}


@torch.no_grad()
def evaluate(model, head, df, device, label_col, criterion_fn, trainable_params, alpha):
    """Evaluate on a split (no gradient)."""
    model.eval()
    head.eval()
    all_logits, all_labels = [], []

    for _, row in df.iterrows():
        volumes = load_and_resample_volume(row['path'])
        channel_embeddings = []
        for volume in volumes:
            volume = volume.to(device)
            cls_token = model(volume)  # (1, 1024)
            channel_embeddings.append(cls_token)
        embedding = torch.cat(channel_embeddings, dim=-1)  # (1, 1024*n_channels)
        logit = head(embedding)  # (1, 1)
        all_logits.append(logit.squeeze(-1).cpu())
        all_labels.append(torch.tensor([row[label_col]], dtype=torch.float32))

    return compute_metrics(all_logits, all_labels, criterion_fn, trainable_params, alpha)


def train_one_epoch(model, head, df, device, label_col, optimizer, criterion_fn,
                    trainable_params, alpha, grad_accum_steps):
    """Train one epoch with gradient accumulation, return metrics."""
    model.train()
    head.train()

    # Shuffle training data
    df = df.sample(frac=1).reset_index(drop=True)

    all_logits, all_labels = [], []
    optimizer.zero_grad()

    for i, (_, row) in enumerate(df.iterrows()):
        volumes = load_and_resample_volume(row['path'])
        channel_embeddings = []
        for volume in volumes:
            volume = volume.to(device)
            cls_token = model(volume)  # (1, 1024)
            channel_embeddings.append(cls_token)
        embedding = torch.cat(channel_embeddings, dim=-1)
        logit = head(embedding).squeeze(-1)  # (1,)
        label = torch.tensor([row[label_col]], dtype=torch.float32, device=device)

        nll = F.binary_cross_entropy_with_logits(logit, label)

        # Regularization on trainable params
        if criterion_fn == 'L1':
            penalty = (alpha / 2) * torch.abs(trainable_params).sum()
        elif criterion_fn == 'L2':
            penalty = (alpha / 2) * (trainable_params ** 2).sum()
        else:
            penalty = torch.tensor(0.0, device=device)

        loss = (nll + penalty) / grad_accum_steps
        loss.backward()

        all_logits.append(logit.detach().cpu())
        all_labels.append(label.detach().cpu())

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(df):
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

    return compute_metrics(all_logits, all_labels, criterion_fn, trainable_params.detach(), alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune 3DINO ViT-Large with LoRA.')
    parser.add_argument('--dataset', type=str, required=True, choices=['oasis3_mri', 'oasis3_ct', 'kpsc'])
    parser.add_argument('--numpy_dir', type=str, required=True)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--hf_download', action='store_true', default=False)
    parser.add_argument('--experiments_dir', type=str, default='')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--criterion', type=str, default='ERM', choices=['ERM', 'L1', 'L2'])
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--label_index', type=int, default=0, help='KPSC: 0=idCBI, 1=idWMD')
    parser.add_argument('--train_site_ids', nargs='+', type=int, default=None)
    parser.add_argument('--val_site_ids', nargs='+', type=int, default=None)
    parser.add_argument('--test_site_ids', nargs='+', type=int, default=None)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.experiments_dir, exist_ok=True)

    # --- Resolve pretrained weights ---
    if args.hf_download:
        from huggingface_hub import hf_hub_download
        args.pretrained_weights = hf_hub_download(
            repo_id="AICONSlab/3DINO-ViT",
            filename="3dino_vit_weights.pth",
        )
        print(f"Downloaded weights to: {args.pretrained_weights}")
    assert args.pretrained_weights is not None, "Provide --pretrained_weights or --hf_download"

    # --- Data splitting ---
    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')

    if args.dataset in ('oasis3_mri', 'oasis3_ct'):
        label_col = "Alzheimer's"
        grouped_df = labels_df.groupby('Subject')[label_col].agg(lambda x: x.mode()[0]).reset_index()
        ids, id_labels = grouped_df['Subject'], grouped_df[label_col]
        train_and_val_ids, test_ids, train_and_val_id_labels, _ = train_test_split(
            ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels
        )
        train_ids, val_ids = train_test_split(
            train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels
        )
        train_df = labels_df[labels_df['Subject'].isin(train_ids)]
        val_df = labels_df[labels_df['Subject'].isin(val_ids)]
        test_df = labels_df[labels_df['Subject'].isin(test_ids)]
    elif args.dataset == 'kpsc':
        label_cols = ['idCBI', 'idWMD']
        label_col = label_cols[args.label_index]
        assert args.train_site_ids and args.val_site_ids and args.test_site_ids, \
            "KPSC requires --train_site_ids, --val_site_ids, --test_site_ids"
        train_df = labels_df[labels_df['SiteID'].isin(args.train_site_ids)]
        val_df = labels_df[labels_df['SiteID'].isin(args.val_site_ids)]
        test_df = labels_df[labels_df['SiteID'].isin(args.test_site_ids)]

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- Load 3DINO model ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_3dino_model(args.pretrained_weights)
    model.to(device)

    # Print module names for LoRA target verification
    print("\n--- Model module names (for LoRA target verification) ---")
    for name, _ in model.named_modules():
        if 'attn' in name.lower():
            print(f"  {name}")
    print("---\n")

    # --- Apply LoRA ---
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Determine embedding dimension ---
    # CLS token only: 1024 per channel
    embed_dim = model.base_model.model.embed_dim
    # Detect number of channels from first training sample
    sample_volumes = load_and_resample_volume(train_df.iloc[0]['path'])
    n_channels = len(sample_volumes)
    in_features = embed_dim * n_channels
    print(f"embed_dim={embed_dim}, n_channels={n_channels}, head in_features={in_features}")

    # --- Classification head ---
    head = nn.Linear(in_features, 1).to(device)

    # --- Trainable params for regularization ---
    def get_trainable_params():
        params = []
        for p in model.parameters():
            if p.requires_grad:
                params.append(p.view(-1))
        for p in head.parameters():
            if p.requires_grad:
                params.append(p.view(-1))
        return torch.cat(params)

    # --- Optimizer (only trainable params) ---
    trainable_param_list = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.SGD(trainable_param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    # --- Training loop ---
    columns = ['epoch', 'test_auroc', 'test_auprc', 'test_bal_acc', 'test_loss', 'test_nll',
               'train_auroc', 'train_auprc', 'train_bal_acc', 'train_loss', 'train_nll',
               'val_auroc', 'val_auprc', 'val_bal_acc', 'val_loss', 'val_nll']
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        trainable_params = get_trainable_params()

        train_metrics = train_one_epoch(
            model, head, train_df, device, label_col, optimizer,
            args.criterion, trainable_params, args.alpha, args.grad_accum_steps
        )

        trainable_params = get_trainable_params()
        val_metrics = evaluate(model, head, val_df, device, label_col, args.criterion, trainable_params, args.alpha)
        test_metrics = evaluate(model, head, test_df, device, label_col, args.criterion, trainable_params, args.alpha)

        row = [epoch,
               test_metrics['auroc'], test_metrics['auprc'], test_metrics['bal_acc'], test_metrics['loss'], test_metrics['nll'],
               train_metrics['auroc'], train_metrics['auprc'], train_metrics['bal_acc'], train_metrics['loss'], train_metrics['nll'],
               val_metrics['auroc'], val_metrics['auprc'], val_metrics['bal_acc'], val_metrics['loss'], val_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])

        model_history_df.to_csv(f'{args.experiments_dir}/{args.model_name}.csv')

        # Save on best val_auroc (where train_auroc > val_auroc to avoid overfitting)
        val_auroc_series = model_history_df[model_history_df.train_auroc > model_history_df.val_auroc].val_auroc
        if args.save and epoch == (val_auroc_series.idxmax() if not val_auroc_series.empty else None):
            save_dir = f'{args.experiments_dir}/{args.model_name}_lora'
            model.save_pretrained(save_dir)
            torch.save(head.state_dict(), f'{args.experiments_dir}/{args.model_name}_head.pt')
            print(f"Saved LoRA adapter to {save_dir} and head to {args.model_name}_head.pt")
