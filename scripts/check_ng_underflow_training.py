"""Check for r_hat underflow DURING actual training (not just at the final selected
checkpoint), starting from random initialization, for NG's fit_std=True mode.
Logs, per epoch: min sigma_fit across sampled bags, and underflow rate.
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import losses  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402

DATA = "/cluster/tufts/hugheslab/datasets"
SEED = 1001
EPOCHS = 60

torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

d_train = torch.load(f"{DATA}/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed={SEED}/train.pt", map_location="cpu", weights_only=False)
train_ds = datasets.MILTensorDataset(d_train["X"], d_train["lengths"], d_train["y"])
loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)

model = models.PoolClf(768, 1, pooling="ABMIL").to(device)
criterion = losses.GuidedAttentionL1Loss(alpha=0.0001, beta=1.0, criterion=torch.nn.BCEWithLogitsLoss(),
                                          divergence="forward kl", fit_mean=True, fit_std=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def check_underflow(model, ds, max_bags=150):
    n_under, n_total, sigmas = 0, 0, []
    with torch.no_grad():
        cnt = 0
        for i in range(len(ds)):
            h_i, S_i, y_i = ds[i]
            if y_i != 1.0:
                continue
            cnt += 1
            if cnt > max_bags:
                break
            h_i = h_i.to(device)
            _, a = model(h_i, (int(S_i),))
            a = a.mean(dim=1).cpu()
            S_i = int(S_i)
            j = torch.arange(1, S_i + 1, dtype=torch.float32)
            sum_a = torch.clamp(a.sum(), min=1e-6)
            mean = (j * a).sum() / sum_a
            var = (j.pow(2) * a).sum() / sum_a - mean.pow(2)
            std = torch.sqrt(torch.clamp(var, min=1e-12))
            sigmas.append(float(std))
            r_hat = utils.normal_pdf(j, mean, std)
            n_under += int((r_hat == 0.0).sum())
            n_total += S_i
    return n_under, n_total, sigmas


model.eval()
n_under, n_total, sigmas = check_underflow(model, train_ds)
print(f"epoch 0 (random init): underflow {n_under}/{n_total} ({100*n_under/max(n_total,1):.2f}%)  min_sigma={min(sigmas):.3f} median_sigma={np.median(sigmas):.3f}", flush=True)

for epoch in range(EPOCHS):
    model.train()
    for images, lengths, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images, lengths)
        logits, attn_weights = out[0], out[1]
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        loss_dict = criterion(logits, labels, attn_weights, lengths, params)
        loss_dict["loss"].backward()
        optimizer.step()

    if epoch < 10 or epoch % 5 == 0:
        model.eval()
        n_under, n_total, sigmas = check_underflow(model, train_ds)
        print(f"epoch {epoch+1}: underflow {n_under}/{n_total} ({100*n_under/max(n_total,1):.2f}%)  min_sigma={min(sigmas):.3f} median_sigma={np.median(sigmas):.3f}", flush=True)
