import copy
# PyTorch
import torch
# Importing our custom module(s)
import utils

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):
        
        nll = self.criterion(logits, labels)
        
        return {'loss': nll, 'nll': nll}
    
class L1Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * torch.abs(params).sum()
        
        return {'loss': nll + penalty, 'nll': nll}
    
class L2Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * (params**2).sum()
        
        return {'loss': nll + penalty, 'nll': nll}

class VAPLoss(torch.nn.Module):
    def __init__(self, beta=1.0, criterion=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.beta = beta
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):
        model = kwargs['model']
        nll = self.criterion(logits, labels)
        kl_loss = model.kl_loss
        total = nll + self.beta * kl_loss
        return {'loss': total, 'nll': nll, 'kl': kl_loss}


class VAPREINFORCELoss(torch.nn.Module):
    def __init__(self, beta=1.0, criterion=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.beta = beta
        self.criterion = criterion
        self._baseline = None
        self._baseline_momentum = 0.9

    def forward(self, logits, labels, **kwargs):
        model = kwargs['model']
        nll = self.criterion(logits, labels)
        kl_loss = model.kl_loss

        # REINFORCE term
        log_prob = model.pool._log_prob  # (N_total, 1)
        task_loss = nll.detach()

        if self._baseline is None:
            self._baseline = task_loss.item()
        advantage = task_loss - self._baseline
        self._baseline = self._baseline_momentum * self._baseline + \
                         (1 - self._baseline_momentum) * task_loss.item()

        reinforce_term = (advantage * log_prob).mean()
        total = nll + self.beta * kl_loss + reinforce_term
        return {'loss': total, 'nll': nll, 'kl': kl_loss}


class GuidedAttentionL1Loss(torch.nn.Module):
    def __init__(self, alpha, beta, criterion=torch.nn.CrossEntropyLoss(), max_std=1000.0, min_std=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.max_std = max_std
        self.min_std = min_std

    def get_x(self, y):
        assert y.dim() == 1, "get_x() expects 1D tensor, got shape {y.shape}"
        return torch.arange(1, len(y) + 1, device=y.device) / len(y)

    def calc_mean(self, y):
        assert y.dim() == 1, "calc_mean() expects 1D tensor, got shape {y.shape}"
        x = self.get_x(y)
        return torch.sum(x * y) / torch.sum(y)

    def calc_std(self, y):
        assert y.dim() == 1, "calc_std() expects 1D tensor, got shape {y.shape}"
        x = self.get_x(y)
        mean = torch.sum(x * y) / torch.sum(y)
        variance = torch.sum((x - mean)**2) / torch.sum(y)
        return torch.sqrt(variance)

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        lengths = kwargs["lengths"]
        attn_weights = kwargs["attn_weights"].view(-1)
        device = attn_weights.device
        
        nll = self.criterion(logits, labels)
        
        with torch.no_grad():
            js = [self.get_x(attn_weights_i) for attn_weights_i in torch.split(attn_weights, lengths)]
            means = [self.calc_mean(attn_weights_i) for attn_weights_i in torch.split(attn_weights, lengths)]
            #stds = [self.calc_std(attn_weights_i) for attn_weights_i in torch.split(attn_weights, lengths)]
            ideal_stds = [self.min_std/length if label == 1.0 else self.max_std/length for label, length in zip(labels, lengths)]
            r_hats = torch.cat([utils.normal_pdf(j, mean, ideal_std) for j, mean, ideal_std in zip(js, means, ideal_stds)])
            rs = torch.cat([r_hat/(r_hat.sum() + 1e-6) for r_hat in torch.split(r_hats, lengths)])
            
        penalty = (self.alpha/2) * torch.abs(params).sum()
        attn_weights_penalty = (self.beta/2) * torch.stack([(diff**2).mean() for diff in torch.split(attn_weights-rs, lengths)]).mean()
        
        return {'loss': nll + penalty + attn_weights_penalty, 'nll': nll}
    
class GuidedNormalL1Loss(torch.nn.Module):
    def __init__(self, alpha, beta, criterion=torch.nn.CrossEntropyLoss(), max_std=100.0, min_std=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.max_std = max_std
        self.min_std = min_std

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        lengths = kwargs["lengths"]
        means, stds = kwargs["attn_weights"]
        device = stds.device
        
        nll = self.criterion(logits, labels)
        
        with torch.no_grad():
            ideal_stds = copy.deepcopy(stds.detach())
            ideal_stds = torch.tensor([max(self.min_std, std-1.0)/length if label == 1.0 else min(self.max_std, std+1.0)/length for std, label, length in zip(stds, labels, lengths)], device=device)
            #ideal_stds = torch.tensor([self.min_std/length if label == 1.0 else self.max_std/length for label, length in zip(labels, lengths)], device=device)
            
        penalty = (self.alpha/2) * torch.abs(params).sum()
        stds_penalty = (self.beta/2) * ((stds-ideal_stds)**2).mean()
        
        return {'loss': nll + penalty + stds_penalty, 'nll': nll}
    