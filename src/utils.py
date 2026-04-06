import math
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F

def inv_sigmoid(x):
    return torch.log(x / (1 - x))

def normal_pdf(x, mu=0.0, sigma=1.0):
    norm_const = 1 / math.sqrt(2.0 * math.pi * sigma**2)
    exp_quad_term = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exp_quad_term

def log_normal_pdf(x, mu=0.0, sigma=1.0):
    log_norm_const = -math.log(math.sqrt(2.0 * math.pi * sigma**2))
    quad_term = -0.5 * ((x - mu) / sigma) ** 2
    return log_norm_const + quad_term

def read_npz(path, key='arr_0', dtype=torch.float32):
    data = np.load(path)
    return torch.as_tensor(data[key], dtype=dtype)

def pad_image(image):
    D, C, H, W = image.shape
    size = max(H, W)
    pad_val = image.min()
    padded_image = torch.full((D, C, size, size), pad_val, dtype=image.dtype, device=image.device)
    padded_image[:,:,(size-H)//2:(size-H)//2+H,(size-W)//2:(size-W)//2+W] = image
    return padded_image

def collate_fn(batch):
    images, lengths, labels = zip(*batch)
    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, lengths, labels
    
def encode_image(model, image):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()

    with torch.no_grad():
        
        if device.type == 'cuda':
            image = image.to(device)
            
        encoded_image = model(image)
        
        if device.type == 'cuda':
            encoded_image = encoded_image.cpu()
            
    return encoded_image

def load_3dino_model(pretrained_weights, dino_repo_path=None):
    """Load 3DINO ViT-Large teacher model.

    Args:
        pretrained_weights: Path to 3dino_vit_weights.pth
        dino_repo_path: Path to 3DINO repo (defaults to ../../3DINO relative to this file)
    """
    if dino_repo_path is None:
        dino_repo_path = '/cluster/tufts/hugheslab/dloevl01/DINO3_Experiments/3DINO'
    sys.path.insert(0, dino_repo_path)

    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

    cfg = load_and_merge_config_3d('train/vit3d_highres')
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    return model


def normalize_volume_3dino(volume):
    """Percentile-based normalization to [-1, 1] (3DINO: 0.05th to 99.95th percentile)."""
    min_val = torch.quantile(volume.float(), 0.0005)
    max_val = torch.quantile(volume.float(), 0.9995)
    volume = (volume - min_val) / (max_val - min_val + 1e-8)
    volume = torch.clip(volume * 2 - 1, -1, 1)
    return volume


def load_and_resample_volume(path, target_size=(112, 112, 112)):
    """Load .npz and resample to target size. Returns list of (1,1,D,H,W) volumes, one per channel."""
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
        volume = torch.rot90(volume, k=1, dims=[2, 3])  # match RAS orientation
        volume = F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
        volume = normalize_volume_3dino(volume)
        volumes.append(volume)
    return volumes


def create_linear_input_3dino(x_tokens_list, use_n_blocks, use_avgpool):
    """Construct features from intermediate layers (matches 3DINO eval/linear3d.py).

    Concatenates CLS tokens from last N blocks, optionally appends
    mean-pooled patch tokens from the final block.
    """
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (output, torch.mean(intermediate_output[-1][0], dim=1)),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


def encode_image_3dino(model, volume, device, n_last_blocks=4, avgpool=True):
    """Encode a single (1, 1, D, H, W) volume with 3DINO and return its embedding."""
    volume = volume.to(device)
    with torch.no_grad():
        if n_last_blocks > 1 or avgpool:
            features = model.get_intermediate_layers(
                volume, n_last_blocks, return_class_token=True
            )
            return create_linear_input_3dino(features, n_last_blocks, avgpool).cpu()
        else:
            return model(volume).cpu()


def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
        
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'auroc': 0.0, 'auprc': 0.0, 'bal_acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}

    for images, lengths, labels in dataloader:
        
        batch_size = len(lengths)

        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        logits, attn_weights = model(images, lengths)
        losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, N=len(dataloader.dataset))
        losses['loss'].backward()
        
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)
            
        optimizer.step()
                
        if lr_scheduler:
            lr_scheduler.step()

        metrics['loss'] += (batch_size / dataset_size) * losses['loss'].item()
        metrics['nll'] += (batch_size / dataset_size) * losses['nll'].item()

        if device.type == 'cuda':
            labels, logits = labels.detach().cpu(), logits.detach().cpu()

        metrics['labels'].extend(labels)
        metrics['logits'].extend(logits)
            
    logits = torch.stack(metrics['logits'])
    probs = torch.nn.functional.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = torch.stack(metrics['labels'])
    metrics['auroc'] = roc_auc_score(labels.numpy(), probs)
    metrics['auprc'] = average_precision_score(labels.numpy(), probs)
    metrics['bal_acc'] = balanced_accuracy_score(labels.numpy(), preds)
    
    return metrics

def evaluate(model, criterion, dataloader):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()
        
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'auroc': 0.0, 'auprc': 0.0, 'bal_acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}

    with torch.no_grad():
        for images, lengths, labels in dataloader:
            
            batch_size = len(lengths)

            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)

            params = torch.nn.utils.parameters_to_vector(model.parameters())
            logits, attn_weights = model(images, lengths)
            losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, N=len(dataloader.dataset))

            metrics['loss'] += (batch_size / dataset_size) * losses['loss'].item()
            metrics['nll'] += (batch_size / dataset_size) * losses['nll'].item()

            if device.type == 'cuda':
                labels, logits = labels.detach().cpu(), logits.detach().cpu()

            metrics['labels'].extend(labels)
            metrics['logits'].extend(logits)

        logits = torch.stack(metrics['logits'])
        probs = torch.nn.functional.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        labels = torch.stack(metrics['labels'])
        metrics['auroc'] = roc_auc_score(labels.numpy(), probs)
        metrics['auprc'] = average_precision_score(labels.numpy(), probs)
        metrics['bal_acc'] = balanced_accuracy_score(labels.numpy(), preds)
            
    return metrics
