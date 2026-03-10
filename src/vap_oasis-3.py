import argparse
import os
import pandas as pd
import torch
# Importing our custom module(s)
import datasets
import losses
import models
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="vap_oasis-3.py")
    parser.add_argument("--batch_size", default=64, help="Batch size (default: 64)", type=int)
    parser.add_argument("--dataset_dir", default="", help="Directory to dataset (default: \"\")", type=str)
    parser.add_argument("--epochs", default=1000, help="Number of epochs (default: 1000)", type=int)
    parser.add_argument("--experiments_dir", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--lr", default=0.01, help="Learning rate (default: 0.01)", type=float)
    parser.add_argument("--model_name", default="test", help="Model name (default: \"test\")", type=str)
    parser.add_argument("--save", action="store_true", default=False, help="Whether or not to save the model (default: False)")
    parser.add_argument("--seed", default=42, help="Random seed (default: 42)", type=int)
    parser.add_argument("--weight_decay", default=0.0, help="Weight decay (default: 0.0)", type=float)
    # VAP-specific arguments
    parser.add_argument("--model_type", default="VAPGaussian", help="VAP model type (default: \"VAPGaussian\")", type=str,
                        choices=["VAPGaussian", "VAPBernoulli", "VAPGaussianSparse"])
    parser.add_argument("--beta", default=0.01, help="KL weight (default: 0.01)", type=float)
    parser.add_argument("--estimator", default="gumbel", help="Bernoulli estimator (default: \"gumbel\")", type=str,
                        choices=["gumbel", "straight_through", "reinforce"])
    parser.add_argument("--pi_0", default=0.1, help="Bernoulli sparsity prior (default: 0.1)", type=float)
    parser.add_argument("--tau_start", default=1.0, help="Initial temperature (default: 1.0)", type=float)
    parser.add_argument("--tau_min", default=0.1, help="Minimum temperature (default: 0.1)", type=float)
    parser.add_argument("--anneal_rate", default=0.95, help="Tau decay rate per epoch (default: 0.95)", type=float)
    parser.add_argument("--prior", default="standard", help="Prior type for VAPGaussian: 'standard' or 'center' (default: 'standard')", type=str,
                        choices=["standard", "center"])
    parser.add_argument("--prior_scale", default=1.0, help="Prior scale (default: 1.0)", type=float)
    parser.add_argument("--adaptive_temp", action="store_true", default=False,
                        help="Use adaptive softmax temperature from learned sigma (default: False)")
    parser.add_argument("--adaptive_temp_scale", default=4.0, help="Scale for adaptive temp (default: 4.0)", type=float)
    parser.add_argument("--mc_samples", default=10, help="Monte Carlo samples for uncertainty (default: 10)", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.experiments_dir, exist_ok=True)

    train_data = torch.load(f"{args.dataset_dir}/train.pth", map_location=torch.device("cpu"), weights_only=False)
    val_data = torch.load(f"{args.dataset_dir}/val.pth", map_location=torch.device("cpu"), weights_only=False)
    test_data = torch.load(f"{args.dataset_dir}/test.pth", map_location=torch.device("cpu"), weights_only=False)

    train_dataset = datasets.MILTensorDataset(train_data["X"], train_data["lengths"], train_data["y"])
    val_dataset = datasets.MILTensorDataset(val_data["X"], val_data["lengths"], val_data["y"])
    test_dataset = datasets.MILTensorDataset(test_data["X"], test_data["lengths"], test_data["y"])

    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)

    in_features = train_data["X"].shape[1]

    if args.model_type == "VAPGaussian":
        model = models.VAPGaussianMIL(in_features=in_features, out_features=1,
                                       prior=args.prior, prior_scale=args.prior_scale,
                                       adaptive_temp=args.adaptive_temp,
                                       adaptive_temp_scale=args.adaptive_temp_scale)
    elif args.model_type == "VAPBernoulli":
        model = models.VAPBernoulliMIL(
            in_features=in_features, out_features=1,
            estimator=args.estimator, pi_0=args.pi_0,
            tau_start=args.tau_start, tau_min=args.tau_min, anneal_rate=args.anneal_rate,
        )
    elif args.model_type == "VAPGaussianSparse":
        model = models.VAPGaussianSparseMIL(
            in_features=in_features, out_features=1,
            prior_scale=args.prior_scale,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    if args.model_type == "VAPBernoulli" and args.estimator == "reinforce":
        criterion = losses.VAPREINFORCELoss(beta=args.beta)
    else:
        criterion = losses.VAPLoss(beta=args.beta)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    columns = [
        "epoch",
        "test_auroc", "test_auprc", "test_bal_acc", "test_loss", "test_nll", "test_kl",
        "train_auroc", "train_auprc", "train_bal_acc", "train_loss", "train_nll", "train_kl",
        "val_auroc", "val_auprc", "val_bal_acc", "val_loss", "val_nll", "val_kl",
    ]
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):

        shuffled_train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
        train_metrics = shuffled_train_metrics
        val_metrics = utils.evaluate(model, criterion, val_loader)
        test_metrics = utils.evaluate(model, criterion, test_loader)

        row = [
            epoch,
            test_metrics["auroc"], test_metrics["auprc"], test_metrics["bal_acc"], test_metrics["loss"], test_metrics["nll"], test_metrics["kl"],
            train_metrics["auroc"], train_metrics["auprc"], train_metrics["bal_acc"], train_metrics["loss"], train_metrics["nll"], train_metrics["kl"],
            val_metrics["auroc"], val_metrics["auprc"], val_metrics["bal_acc"], val_metrics["loss"], val_metrics["nll"], val_metrics["kl"],
        ]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])

        model_history_df.to_csv(f"{args.experiments_dir}/{args.model_name}.csv")

        # Temperature annealing for Bernoulli models
        if hasattr(model.pool, 'anneal_temperature'):
            model.pool.anneal_temperature()

        val_auroc_series = model_history_df[model_history_df.train_auroc > model_history_df.val_auroc].val_auroc
        if args.save and epoch == (val_auroc_series.idxmax() if not val_auroc_series.empty else None):
            torch.save(model.state_dict(), f"{args.experiments_dir}/{args.model_name}.pt")
