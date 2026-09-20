"""Train Best-in-Class Ceiling (InstanceLevelClassifier, kernel_size=r=12) on the
Semi-Synthetic dataset, matching the same ShiftedMeanMILDataset config used everywhere
else in this session's Semi-Synthetic column (s_low=20, s_high=60, delta=0.5, r=12),
generated in-memory (no pre-saved .pt files needed, unlike src/best_possible_instance-level.py).

Usage: python scripts/train_ceiling_semi.py --alpha=0.0001 --lr=0.01 --seed=1001 --epochs=50 [--save]
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import datasets  # noqa: E402
import losses  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402

R = 12
S_LOW, S_HIGH, DELTA = 20, 60, 0.5
N_TRAIN, N_VAL, N_TEST = 10000, 2500, 1000

SEED_OFFSETS = {1001: (1001, 1002, 1003), 2001: (2001, 2002, 2003), 3001: (3001, 3002, 3003)}


def build(n, data_seed):
    ds = datasets.ShiftedMeanMILDataset(n=n, r=R, s_low=S_LOW, s_high=S_HIGH, delta=DELTA, seed=data_seed)
    lengths_y = []
    for i, S in enumerate(ds.lengths):
        y_ij = torch.zeros(int(S), 1)
        if ds.y[i] == 1:
            u = int(ds.u[i])
            y_ij[u:min(u + ds.r, int(S)), 0] = 1.0
        lengths_y.append(y_ij)
    return datasets.MILTensorDataset(ds.h, ds.lengths, lengths_y)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--experiments_dir", type=str, default="/cluster/home/zmou01/pooling/experiments/synthetic_best_possible_instance-level")
    p.add_argument("--model_name", type=str, default="test")
    p.add_argument("--save", action="store_true", default=False)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.experiments_dir, exist_ok=True)

    seed_train, seed_val, seed_test = SEED_OFFSETS[args.seed]
    train_dataset = build(N_TRAIN, seed_train)
    val_dataset = build(N_VAL, seed_val)
    test_dataset = build(N_TEST, seed_test)

    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.instance_level_collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=utils.instance_level_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=utils.instance_level_collate_fn)

    model = models.InstanceLevelClassifier(in_features=768, out_features=1, kernel_size=R)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    model.to(device)

    criterion = losses.L1Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0, momentum=0.9)

    columns = ["epoch", "test_auroc", "test_auprc", "test_bal_acc", "test_loss", "test_nll", "train_auroc", "train_auprc", "train_bal_acc", "train_loss", "train_nll", "train_sec/epoch", "val_auroc", "val_auprc", "val_bal_acc", "val_loss", "val_nll"]
    df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
        t1 = time.time()
        val_metrics = utils.evaluate(model, criterion, val_loader)
        test_metrics = utils.evaluate(model, criterion, test_loader)
        row = [epoch, test_metrics["auroc"], test_metrics["auprc"], test_metrics["bal_acc"], test_metrics["loss"], test_metrics["nll"], train_metrics["auroc"], train_metrics["auprc"], train_metrics["bal_acc"], train_metrics["loss"], train_metrics["nll"], t1 - t0, val_metrics["auroc"], val_metrics["auprc"], val_metrics["bal_acc"], val_metrics["loss"], val_metrics["nll"]]
        df.loc[epoch] = row
        print(f"epoch {epoch}: train_auroc={train_metrics['auroc']:.4f} val_auroc={val_metrics['auroc']:.4f} test_auroc={test_metrics['auroc']:.4f} sec/epoch={t1-t0:.3f}", flush=True)
        df.to_csv(f"{args.experiments_dir}/{args.model_name}.csv")
        val_series = df[df.train_auroc > df.val_auroc].val_auroc
        if args.save and epoch == (val_series.idxmax() if not val_series.empty else None):
            torch.save(model.state_dict(), f"{args.experiments_dir}/{args.model_name}.pt")
