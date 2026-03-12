"""
Feasibility check: Load 3DINO-ViT on CPU, run forward pass on dummy data,
save embeddings in MIL-compatible format, and verify pipeline compatibility.

Run from pooling/src/:
    XFORMERS_DISABLED=1 python check_3dino_feasibility.py
"""

import os
import sys
import tempfile

import torch

# Add 3DINO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '3DINO'))

os.environ['XFORMERS_DISABLED'] = '1'  # Force fallback to standard attention


def load_3dino_model_cpu(pretrained_weights=None):
    """Load 3DINO ViT-Large model on CPU."""
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

    cfg = load_and_merge_config_3d('train/vit3d_highres')
    model, _ = build_model_from_cfg(cfg, only_teacher=True)

    if pretrained_weights is not None:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    else:
        print("No pretrained weights provided, using random initialization.")

    model.eval()
    return model


def normalize_volume(volume):
    """Percentile-based normalization to [-1, 1] (as in 3DINO notebook)."""
    min_val = torch.quantile(volume, 0.0005)
    max_val = torch.quantile(volume, 0.9995)
    volume = (volume - min_val) / (max_val - min_val)
    volume = torch.clip(volume * 2 - 1, -1, 1)
    return volume


if __name__ == '__main__':
    print("=" * 60)
    print("3DINO Feasibility Check")
    print("=" * 60)

    # --- Step 1: Load model ---
    print("\n[Step 1] Loading 3DINO ViT-Large model on CPU...")

    # Try to download weights from HuggingFace
    pretrained_weights = None
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading pretrained weights from HuggingFace...")
        pretrained_weights = hf_hub_download(
            repo_id="AICONSlab/3DINO-ViT",
            filename="3dino_vit_weights.pth",
        )
        print(f"Weights downloaded to: {pretrained_weights}")
    except Exception as e:
        print(f"Could not download weights: {e}")
        print("Proceeding with random initialization (forward pass shape check still valid).")

    model = load_3dino_model_cpu(pretrained_weights)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {total_params:,}")
    print(f"Embed dim: {model.embed_dim}")

    # --- Step 2: Create dummy 3D volume ---
    print("\n[Step 2] Creating dummy 3D volume (1, 1, 112, 112, 112)...")
    dummy_volume = torch.randn(1, 1, 112, 112, 112)
    dummy_volume = normalize_volume(dummy_volume)
    print(f"Volume range: [{dummy_volume.min():.2f}, {dummy_volume.max():.2f}]")

    # --- Step 3: Forward pass ---
    print("\n[Step 3] Running forward pass...")
    with torch.no_grad():
        output = model(dummy_volume)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1024), f"Expected (1, 1024), got {output.shape}"
    print("PASS: Output shape is (1, 1024) as expected.")

    # --- Step 4: Save dummy embeddings in MIL-compatible format ---
    print("\n[Step 4] Saving dummy embeddings in {X, lengths, y} format...")
    n_subjects = 10
    embeddings = []
    with torch.no_grad():
        for i in range(n_subjects):
            vol = normalize_volume(torch.randn(1, 1, 112, 112, 112))
            emb = model(vol)  # (1, 1024)
            embeddings.append(emb)

    X = torch.cat(embeddings, dim=0)  # (n_subjects, 1024)
    lengths = tuple([1] * n_subjects)  # Each subject = 1 instance
    y = torch.randint(0, 2, (n_subjects, 1)).float()  # Dummy labels

    tmpdir = tempfile.mkdtemp()
    for split_name in ['train', 'val', 'test']:
        save_path = os.path.join(tmpdir, f'{split_name}.pth')
        torch.save({'X': X, 'lengths': lengths, 'y': y}, save_path)
        print(f"Saved {split_name}.pth: X={X.shape}, lengths={lengths}, y={y.shape}")

    # --- Step 5: Load into MIL pipeline and verify compatibility ---
    print("\n[Step 5] Loading into MIL pipeline with Mean pooling...")
    sys.path.insert(0, os.path.dirname(__file__))
    import datasets
    import models
    import utils

    train_data = torch.load(os.path.join(tmpdir, 'train.pth'), map_location='cpu', weights_only=False)
    train_dataset = datasets.MILTensorDataset(train_data['X'], train_data['lengths'], train_data['y'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, collate_fn=utils.collate_fn
    )

    # Test with PoolClf (embedding-level) + Mean pooling
    mil_model = models.PoolClf(in_features=1024, out_features=1, pooling='Mean')
    mil_model.eval()

    for images, batch_lengths, labels in train_loader:
        logits, attn_weights = mil_model(images, batch_lengths)
        print(f"Batch: images={images.shape}, lengths={batch_lengths}, labels={labels.shape}")
        print(f"Logits: {logits.shape}, attn_weights type: {type(attn_weights)}")
        break

    print("\nPASS: MIL pipeline compatibility verified.")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FEASIBILITY CHECK COMPLETE")
    print("=" * 60)
    print(f"  Model:          3DINO ViT-Large (vit_large_3d)")
    print(f"  Input shape:    (B, 1, 112, 112, 112)")
    print(f"  Output shape:   (B, 1024)")
    print(f"  Weights:        {'Loaded from HuggingFace' if pretrained_weights else 'Random init'}")
    print(f"  xformers:       Disabled (CPU fallback)")
    print(f"  MIL compat:     YES (lengths=1 per subject, Mean pooling = identity)")
    print(f"  Next step:      Write encode_oasis-3_3dino.py for actual data")
