import argparse
import os
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
# Importing our custom module(s)
import losses
import models

def collate_fn(batch):
    bags, lengths, labels, instance_labels = zip(*batch)
    return torch.cat(bags), lengths, torch.stack(labels), torch.cat(instance_labels)

class MILTensorDatasetWithInstanceLabels(torch.utils.data.Dataset):
    def __init__(self, X, lengths, y, lengths_y):
        self.bags = list(torch.split(X, list(lengths)))
        self.lengths = lengths
        self.y = y
        self.instance_labels = [torch.tensor(ly, dtype=torch.float32).unsqueeze(-1) for ly in lengths_y]

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.lengths[idx], self.y[idx], self.instance_labels[idx]

def train_one_epoch(model, criterion, optimizer, dataloader):
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'auroc': 0.0, 'auprc': 0.0, 'bal_acc': 0.0, 'labels': [], 'logits': [], 'loss': 0.0, 'nll': 0.0}

    for images, lengths, labels, instance_labels in dataloader:
        batch_size = len(lengths)
        if device.type == 'cuda':
            images, labels, instance_labels = images.to(device), labels.to(device), instance_labels.to(device)

        optimizer.zero_grad()
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        logits, attn_weights = model(images, lengths)
        loss_dict = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, instance_labels=instance_labels, N=len(dataloader.dataset))
        loss_dict['loss'].backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=1.0)
        optimizer.step()

        metrics['loss'] += (batch_size / dataset_size) * loss_dict['loss'].item()
        metrics['nll'] += (batch_size / dataset_size) * loss_dict['nll'].item()

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
        for images, lengths, labels, instance_labels in dataloader:
            batch_size = len(lengths)
            if device.type == 'cuda':
                images, labels, instance_labels = images.to(device), labels.to(device), instance_labels.to(device)

            params = torch.nn.utils.parameters_to_vector(model.parameters())
            logits, attn_weights = model(images, lengths)
            loss_dict = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, instance_labels=instance_labels, N=len(dataloader.dataset))

            metrics['loss'] += (batch_size / dataset_size) * loss_dict['loss'].item()
            metrics['nll'] += (batch_size / dataset_size) * loss_dict['nll'].item()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rsna_supervised_vap.py')
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--criterion', default='ERM', type=str)
    parser.add_argument('--dataset_dir', default='', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--experiments_dir', default='', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--model_name', default='test', type=str)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--pi_max', default=0.5, type=float)
    parser.add_argument('--sigma', default=0.25, type=float)
    parser.add_argument('--tau', default=0.5, type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.experiments_dir, exist_ok=True)

    train_data = torch.load(f'{args.dataset_dir}/train.pth', map_location='cpu', weights_only=False)
    val_data = torch.load(f'{args.dataset_dir}/val.pth', map_location='cpu', weights_only=False)
    test_data = torch.load(f'{args.dataset_dir}/test.pth', map_location='cpu', weights_only=False)

    train_dataset = MILTensorDatasetWithInstanceLabels(train_data['X'], train_data['lengths'], train_data['y'], train_data['lengths_y'])
    val_dataset = MILTensorDatasetWithInstanceLabels(val_data['X'], val_data['lengths'], val_data['y'], val_data['lengths_y'])
    test_dataset = MILTensorDatasetWithInstanceLabels(test_data['X'], test_data['lengths'], test_data['y'], test_data['lengths_y'])

    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    pool_kwargs = dict(pi_max=args.pi_max, sigma=args.sigma, tau=args.tau)
    model = models.PoolClf(in_features=train_data['X'].shape[1], out_features=1, pooling='BernoulliVAP', **pool_kwargs)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    assert args.criterion in ['ERM', 'L1', 'L2']
    if args.criterion == 'ERM':
        base_criterion = losses.ERMLoss(criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == 'L1':
        base_criterion = losses.L1Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == 'L2':
        base_criterion = losses.L2Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())

    criterion = losses.SupervisedVAPLoss(base_criterion, beta=args.beta)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    columns = ['epoch', 'test_auroc', 'test_auprc', 'test_bal_acc', 'test_loss', 'test_nll', 'train_auroc', 'train_auprc', 'train_bal_acc', 'train_loss', 'train_nll', 'val_auroc', 'val_auprc', 'val_bal_acc', 'val_loss', 'val_nll']
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        shuffled_train_metrics = train_one_epoch(model, criterion, optimizer, shuffled_train_loader)
        train_metrics = shuffled_train_metrics
        val_metrics = evaluate(model, criterion, val_loader)
        test_metrics = evaluate(model, criterion, test_loader)

        row = [epoch, test_metrics['auroc'], test_metrics['auprc'], test_metrics['bal_acc'], test_metrics['loss'], test_metrics['nll'], train_metrics['auroc'], train_metrics['auprc'], train_metrics['bal_acc'], train_metrics['loss'], train_metrics['nll'], val_metrics['auroc'], val_metrics['auprc'], val_metrics['bal_acc'], val_metrics['loss'], val_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])

        model_history_df.to_csv(f'{args.experiments_dir}/{args.model_name}.csv')

        val_auroc_series = model_history_df[model_history_df.train_auroc > model_history_df.val_auroc].val_auroc
        if args.save and epoch == (val_auroc_series.idxmax() if not val_auroc_series.empty else None):
            torch.save(model.state_dict(), f'{args.experiments_dir}/{args.model_name}.pt')
