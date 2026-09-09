import copy
from typing import Dict, Tuple
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

class GuidedAttentionL1Loss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        beta: float,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        eps: float = 1e-6,
        divergence: str = "squared error",
        fit_mean: bool = True,
        fit_std: bool = True,
        empirical_mean: float = 0.5,
        empirical_std: float = 0.15,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.eps = eps
        assert divergence in ["squared error", "forward kl", "reverse kl"]
        self.divergence = divergence
        # Mean/variance ablation: when a moment is not fit from the current
        # attention, the guiding Normal uses a fixed value derived from the
        # empirical distribution of positive-slice positions (normalized to
        # [0, 1] and rescaled per bag by S_i).
        self.fit_mean = fit_mean
        self.fit_std = fit_std
        self.empirical_mean = empirical_mean
        self.empirical_std = empirical_std

    def _bag_mean(self, a: torch.Tensor) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        if self.fit_mean:
            return self._compute_mean(a)
        S_i, num_heads = a.shape
        return torch.full(
            (num_heads,), self.empirical_mean * S_i, device=a.device, dtype=a.dtype
        )

    def _bag_std(self, a: torch.Tensor) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        if self.fit_std:
            return self._compute_std(a)
        S_i, num_heads = a.shape
        return torch.full(
            (num_heads,),
            max(self.empirical_std, self.eps) * S_i,
            device=a.device,
            dtype=a.dtype,
        )
        
    def _get_j(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads]
        return torch.arange(1, len(a) + 1, device=a.device, dtype=a.dtype).unsqueeze(1)

    def _compute_mean(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        j = self._get_j(a)
        sum_a = torch.clamp(a.sum(dim=0), min=self.eps)
        return (j * a).sum(dim=0) / sum_a

    def _compute_std(
        self,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # a shape: [S_i, num_heads] -> [num_heads]
        j = self._get_j(a)
        sum_a = torch.clamp(a.sum(dim=0), min=self.eps)
        mean = (j * a).sum(dim=0) / sum_a
        variance = torch.clamp(
            (j ** 2 * a).sum(dim=0) / sum_a - mean ** 2,
            min=self.eps,
        )
        return variance.sqrt()

    def _compute_divergence(
        self,
        r: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        if self.divergence == "squared error":
            return (r - a) ** 2
        elif self.divergence == "forward kl":
            return r * (r.log() - a.log())
        elif self.divergence == "reverse kl":
            return a * (a.log() - r.log())

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attn_weights: torch.Tensor,
        lengths: Tuple[int, ...],
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)

        with torch.no_grad():
            
            split_attn_weights = torch.split(attn_weights, lengths, dim=0)
            
            js = [self._get_j(a) for a in split_attn_weights]
            means = [self._bag_mean(a) for a in split_attn_weights]
            stds = [self._bag_std(a) for a in split_attn_weights]

            r_hats = torch.cat([
                utils.normal_pdf(j, mean, std)
                for j, mean, std in zip(js, means, stds)
            ], dim=0)
            rs = torch.cat([
                r_hat / torch.clamp(r_hat.sum(dim=0), min=self.eps)
                for r_hat in torch.split(r_hats, lengths, dim=0)
            ], dim=0)

        penalty = (self.alpha / 2) * params.abs().sum()

        rs = torch.clamp(rs, min=self.eps)
        attn_weights = torch.clamp(attn_weights, min=self.eps)

        diff = self._compute_divergence(rs, attn_weights)

        attn_penalty = self.beta * torch.stack([
            diff_i.sum(dim=0).mean()
            for diff_i in torch.split(diff, lengths, dim=0)
        ]).mean()

        return {
            "loss": nll + penalty + attn_penalty, 
            "nll": nll,
        }
    
class AEML1Loss(torch.nn.Module):
    def __init__(
        self,
        alpha: float,
        beta: float,
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attn_weights: torch.Tensor,
        lengths: Tuple[int, ...],
        params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        nll = self.criterion(logits, labels)

        split_attn_weights = torch.split(attn_weights, lengths, dim=0)

        entropy_penalties = []
        for a in split_attn_weights:
            a_clamped = torch.clamp(a, min=self.eps)
            neg_entropy = (a_clamped * a_clamped.log()).sum(dim=0)
            entropy_penalties.append(neg_entropy.mean())

        penalty = (self.alpha / 2) * params.abs().sum()
        aem_penalty = self.beta * torch.stack(entropy_penalties).mean()
        
        return {
            "loss": nll + aem_penalty + penalty,
            "nll": nll,
        }
    