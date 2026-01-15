import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
import torchvision
# Importing our custom module(s)
import datasets
import losses
import models
import utils

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
        logits = model(images)
        losses = criterion(logits, labels, params=params, N=len(dataloader.dataset))
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
            logits = model(images)
            losses = criterion(logits, labels, params=params, N=len(dataloader.dataset))

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

# python ../src/fine-tune_oasis-3.py --alpha=0.0 --batch_size=2 --criterion='ERM' --dataset_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --epochs=100 --experiments_dir='/cluster/tufts/hugheslab/eharve06/pooling/experiments/test' --lr=0.01 --num_workers=0 --model_name='test' --seed=42 --weight_decay=0.0
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--alpha', default=0.0, help='TODO (default: 0.0)', type=float)
    parser.add_argument('--batch_size', default=64, help='Batch size (default: 64)', type=int)
    parser.add_argument('--criterion', default='ERM', help='TODO (default: \'ERM\')', type=str)
    parser.add_argument('--dataset_dir', default='', help='Directory to dataset (default: \'\')', type=str)
    parser.add_argument('--epochs', default=1000, help='Number of epochs (default: 1000)', type=int)
    parser.add_argument('--experiments_dir', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--lr', default=0.01, help='Learning rate (default: 0.01)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--num_workers', default=0, help='Number of workers (default: 0)', type=int)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--seed', default=42, help='TODO (default: 42)', type=int)
    parser.add_argument('--weight_decay', default=0.0, help='Weight decay (default: 0.0)', type=float)
    args = parser.parse_args()
    
    os.makedirs(args.experiments_dir, exist_ok=True)

    labels_df = pd.read_csv(f'{args.dataset_dir}/labels.csv')

    grouped_df = labels_df.groupby('Subject')['Alzheimer\'s'].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df['Subject'], grouped_df['Alzheimer\'s']
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)
    
    train_df = labels_df[labels_df['Subject'].isin(train_ids)]
    val_df = labels_df[labels_df['Subject'].isin(val_ids)]
    test_df = labels_df[labels_df['Subject'].isin(test_ids)]
    
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        #lambda image: image.permute(3, 0, 1, 2),
        lambda image: image.permute(0, 3, 1, 2),
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(224, 224)),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['Alzheimer\'s']].values, dtype=torch.float32), transform)
    
    means, stds = [], []

    for image, num_slices, label in train_dataset:
        means.append(torch.mean(image, dim=(1, 2, 3)).tolist())
        stds.append(torch.std(image, dim=(1, 2, 3)).tolist())

    mean = torch.tensor(means).mean(dim=0)
    std = torch.tensor(stds).mean(dim=0)

    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        #lambda image: image.permute(3, 0, 1, 2),
        lambda image: image.permute(0, 3, 1, 2),
        lambda image: utils.pad_image(image),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
        torchvision.transforms.Resize(size=(224, 224)),
        #lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
        lambda image: (image - mean.view(-1, 1, 1, 1)) / std.view(-1, 1, 1, 1),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['Alzheimer\'s']].values, dtype=torch.float32), transform)
    val_dataset = datasets.MILPathDataset(val_df.path.values, torch.tensor(val_df[['Alzheimer\'s']].values, dtype=torch.float32), transform)
    test_dataset = datasets.MILPathDataset(test_df.path.values, torch.tensor(test_df[['Alzheimer\'s']].values, dtype=torch.float32), transform)
    
    shuffled_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = models.OnTheDesign(num_classes=1)
    #model = torchvision.models.video.r3d_18()
    #model.stem[0].weight.data = model.stem[0].weight.data.sum(dim=1, keepdim=True).repeat(1, 2, 1, 1, 1)
    #model.stem[0].in_channels = 2
    #model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    assert args.criterion in ['ERM', 'L1', 'L2']
    if args.criterion == 'ERM':
        criterion = losses.ERMLoss(criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == 'L1':
        criterion = losses.L1Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == 'L2':
        criterion = losses.L2Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    columns = ['epoch', 'test_auroc', 'test_auprc', 'test_bal_acc', 'test_loss', 'test_nll', 'train_auroc', 'train_auprc', 'train_bal_acc', 'train_loss', 'train_nll', 'val_auroc', 'val_auprc', 'val_bal_acc', 'val_loss', 'val_nll']
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        shuffled_train_metrics = train_one_epoch(model, criterion, optimizer, shuffled_train_dataloader)
        #train_metrics = utils.evaluate(model, criterion, train_dataloader)
        train_metrics = shuffled_train_metrics
        val_metrics = evaluate(model, criterion, val_dataloader)
        test_metrics = evaluate(model, criterion, test_dataloader)
        
        row = [epoch, test_metrics['auroc'], test_metrics['auprc'], test_metrics['bal_acc'], test_metrics['loss'], test_metrics['nll'], train_metrics['auroc'], train_metrics['auprc'], train_metrics['bal_acc'], train_metrics['loss'], train_metrics['nll'], val_metrics['auroc'], val_metrics['auprc'], val_metrics['bal_acc'], val_metrics['loss'], val_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_dir}/{args.model_name}.csv')
    
        if args.save and epoch == model_history_df.val_auroc.idxmax():
            torch.save(model.state_dict(), f'{args.experiments_dir}/{args.model_name}.pt')
            