# PyTorch
import torch
# Importing our custom module(s)
import utils

class AttentionBasedPooling(torch.nn.Module):
    def __init__(self, in_features, temp=1.0):
        super().__init__()
        self.temp = temp
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x, lengths):
        
        attn_logits = self.mlp(x)
        attn_weights = torch.cat([torch.nn.functional.softmax(weights_i/self.temp, dim=0) for weights_i in torch.split(attn_logits, lengths)])
        attn_weighted_x = attn_weights*x
        context_vectors = torch.cat([torch.sum(attn_weighted_x_i, dim=0, keepdim=True) for attn_weighted_x_i in torch.split(attn_weighted_x, lengths)])
        
        return context_vectors, attn_weights
    
class Inflate(torch.nn.Module):
    def __init__(self, input_instances=3):
        super().__init__()
        self.input_instances = input_instances
        self.half_input_instances = int(input_instances/2)

    def forward(self, x, lengths):
        num_instances, hidden_dim = x.shape
        x = torch.cat([torch.nn.functional.pad(x_i, (0, 0, self.half_input_instances, self.half_input_instances), mode='constant', value=0.0) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + (2 * self.half_input_instances) for length in lengths)
        x = torch.cat([x_i.unfold(0, self.input_instances, 1) for x_i in torch.split(x, lengths)])
        x = x.reshape(num_instances, hidden_dim * self.input_instances)
        return x
    
class InstanceConv1d(torch.nn.Module):
    def __init__(self, in_features, kernel_size=3):
        super().__init__()
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=self.kernel_size, groups=self.in_features, padding="same")
        
    def forward(self, x, lengths):
        x = torch.cat([self.conv(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        return x
    
class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        out = torch.cat([torch.max(x_i, dim=0, keepdim=True).values for x_i in torch.split(x, lengths)])
        attn_weights = None
        
        return out, attn_weights  
    
class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        device = x.device        
        out = torch.cat([torch.mean(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
        attn_weights = torch.cat([torch.ones(size=(length, 1), device=device) / length for length in lengths])
        
        return out, attn_weights
    
class NormalPooling(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=2),
        )
        
    def get_x(self, y):
        assert y.dim() == 1, "get_x() expects 1D tensor, got shape {y.shape}"
        return torch.arange(1, len(y) + 1, device=y.device) / len(y)

    def calc_mean(self, y):
        assert y.dim() == 1, "calc_mean() expects 1D tensor, got shape {y.shape}"
        max_y = torch.max(y)
        w = torch.exp(y - max_y)
        x = self.get_x(y)
        return torch.sum(x * w) / torch.sum(w)

    def forward(self, x, lengths):
        
        means_and_stds = self.mlp(x)
        means = torch.stack([self.calc_mean(mean_i) for mean_i in torch.split(means_and_stds[:,0], lengths)])
        stds = torch.stack([torch.nn.functional.softplus(std_i.mean()) for std_i in torch.split(means_and_stds[:,1], lengths)])
        weights = torch.cat([utils.normal_pdf(self.get_x(mean_i), means[i], stds[i]) for i, mean_i in enumerate(torch.split(means_and_stds[:,0], lengths))])
        attn_weights = torch.cat([weights_i/(torch.sum(weights_i) + 1e-3) for weights_i in torch.split(weights, lengths)]).view(-1, 1)
        x = torch.stack([torch.sum(context_vector_i, dim=0) for context_vector_i in torch.split(attn_weights*x, lengths)])
        
        return x, attn_weights

class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=1, temp=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.temp = temp
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
    
    def forward(self, x, lengths):
        
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # First transformer layer
        out, attn_weights = zip(*[self.self_attn(x_i / self.temp**0.5, x_i / self.temp**0.5, x_i) for x_i in torch.split(x, lengths)])
        x = torch.cat(out)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])        
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        return x, attn_weights
    
class PPEG(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.proj1 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=3, groups=self.in_features, padding="same")
        self.proj2 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=5, groups=self.in_features, padding="same")
        self.proj3 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=7, groups=self.in_features, padding="same")

    def forward(self, x, lengths):
        
        # Split class tokens and features
        cls_token = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        x = torch.cat([x_i[1:,:] for x_i in torch.split(x, lengths)])
        lengths = tuple(length - 1 for length in lengths)
        # Reshape patch tokens into 1D image space
        # Use different sized convolutions
        proj1 = torch.cat([self.proj1(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj2 = torch.cat([self.proj2(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj3 = torch.cat([self.proj3(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        # Fuse together different spatial information
        x = x + proj1 + proj2 + proj3
        x = torch.cat([torch.cat((cls_token[i][None,:], x_i)) for i, x_i in enumerate(torch.split(x, lengths))])

        return x

class PositionalEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        
        device = x.device
        positional_encoding = torch.cat([torch.arange(1, length + 1, device=device) / length for length in lengths])
        Phi = torch.cat([positional_encoding.unsqueeze(1), x], dim=1)
        
        return Phi
    
class TransformerLayer(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.norm = torch.nn.LayerNorm(normalized_shape=self.in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
        
    def forward(self, x, lengths):
        out, attn_weights = zip(*[self.attn(x_i, x_i, x_i) for x_i in torch.split(self.norm(x), lengths)])
        x = x + torch.cat(out)
        return x, attn_weights
    
class TransformerBasedPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.layer1 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)
        self.position_layer = PPEG(in_features=self.in_features)        
        self.layer2 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)

    def forward(self, x, lengths):
        
        device = x.device
        # Concatenate class token
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # Add positional encoding
        positional_encoding = torch.cat([torch.arange(0, length, device=device) / length for length in lengths])        
        x = x + positional_encoding[:,None]
        # First transformer layer
        x, _ = self.layer1(x, lengths)
        # Pyramid position encoding generator layer
        x = self.position_layer(x, lengths)
        # Second transformer layer
        x, attn_weights = self.layer2(x, lengths)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])

        return x, attn_weights


class ApproxSm(torch.nn.Module):
    def __init__(self, alpha=0.5, num_steps=10, learnable_alpha=True):
        super().__init__()
        self.num_steps = int(num_steps)
        self.learnable_alpha = bool(learnable_alpha)
        if self.learnable_alpha:
            self.raw_alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha / (1 - alpha), dtype=torch.float32)))
        else:
            self.register_buffer("raw_alpha", torch.log(torch.tensor(alpha / (1 - alpha), dtype=torch.float32)))

    def _alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def forward(self, f: torch.Tensor, neighbors: int = 1, self_loop: bool = False) -> torch.Tensor:
        alpha = self._alpha()
        S = f.size(0)
        if S <= 1 or neighbors <= 0:
            return f.clone()
        squeeze = f.dim() == 1
        if squeeze:
            f = f.unsqueeze(1)
        g = f.clone()
        for _ in range(self.num_steps):
            Ag = self._neighbor_average(g, neighbors, self_loop)
            g = (1.0 - alpha) * f + alpha * Ag
        return g.squeeze(1) if squeeze else g

    def _neighbor_average(self, g: torch.Tensor, radius: int, self_loop: bool) -> torch.Tensor:
        S, d = g.shape
        agg = torch.zeros_like(g)
        deg = torch.zeros((S, 1), device=g.device, dtype=g.dtype)
        if self_loop:
            agg += g
            deg += 1.0
        for k in range(1, radius + 1):
            agg[k:] += g[:-k]
            deg[k:] += 1.0
            agg[:-k] += g[k:]
            deg[:-k] += 1.0
        return agg / deg.clamp_min(1e-12)


def _build_chain_A(length: int, radius: int = 1, self_loop: bool = False, *, device=None, dtype=None):
    A = torch.zeros((length, length), device=device, dtype=dtype)
    if self_loop:
        A.fill_diagonal_(1.0)
    if length >= 2:
        for k in range(1, radius + 1):
            A[k:, :-k] += torch.eye(length - k, device=device, dtype=dtype)
            A[:-k, k:] += torch.eye(length - k, device=device, dtype=dtype)
    rowsum = A.sum(dim=1, keepdim=True).clamp_min(1e-12)
    A = A / rowsum
    return A


class ExactSm(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        if alpha == 'trainable':
            self._raw = torch.nn.Parameter(torch.log(torch.tensor(alpha / (1 - alpha), dtype=torch.float32)))
            self.register_buffer('_alpha_fixed', None)
        else:
            a = float(alpha)
            if not (0.0 <= a < 1.0):
                raise ValueError("alpha must be in [0,1).")
            self._raw = None
            self.register_buffer('_alpha_fixed', torch.tensor(a))

    def _alpha(self):
        return torch.sigmoid(self._raw) if self._raw is not None else self._alpha_fixed

    def forward(self, f: torch.Tensor, neighbors: int = 1, self_loop: bool = False) -> torch.Tensor:
        a = self._alpha()
        S = f.size(0)
        if S <= 1:
            return f.clone()
        squeeze = False
        if f.dim() == 1:
            f2 = f.unsqueeze(1)
            squeeze = True
        else:
            f2 = f
        A = _build_chain_A(S, radius=neighbors, self_loop=self_loop, device=f.device, dtype=f.dtype)
        M = torch.eye(S, device=f.device, dtype=f.dtype) - a * A
        rhs = (1.0 - a) * f2
        g2 = torch.linalg.solve(M, rhs)
        return g2.squeeze(1) if squeeze else g2


class SmMILPooling(torch.nn.Module):
    def __init__(self, in_features, temp=1.0, sm_alpha=0.5, sm_steps=10, sm_where='early'):
        super().__init__()
        self.in_features = in_features
        self.temp = temp
        self.sm_alpha = sm_alpha
        self.sm_steps = sm_steps
        self.sm_where = sm_where
        fc1 = torch.nn.Linear(in_features=in_features, out_features=128)
        fc2 = torch.nn.Linear(in_features=128, out_features=1)
        if sm_where == 'early':
            fc1 = torch.nn.utils.parametrizations.spectral_norm(fc1)
            fc2 = torch.nn.utils.parametrizations.spectral_norm(fc2)
        self.mlp = torch.nn.Sequential(fc1, torch.nn.Tanh(), fc2)
        self.sm_layer_approx = ApproxSm(alpha=self.sm_alpha, num_steps=self.sm_steps, learnable_alpha=True)

    def forward(self, x, lengths, neighbors=1):
        if self.sm_where == 'early':
            x_smoothed = torch.cat([
                self.sm_layer_approx(x_i, neighbors=neighbors, self_loop=False)
                for x_i in torch.split(x, lengths)
            ])
            attn_logits = self.mlp(x_smoothed)
        else:
            attn_logits = self.mlp(x)
            attn_logits = torch.cat([
                self.sm_layer_approx(logits_i, neighbors=neighbors, self_loop=False)
                for logits_i in torch.split(attn_logits, lengths)
            ])
        attn_weights = torch.cat([
            torch.nn.functional.softmax(logits_i / self.temp, dim=0)
            for logits_i in torch.split(attn_logits, lengths)
        ])
        attn_weighted_x = attn_weights * x
        context_vectors = torch.cat([
            torch.sum(attn_weighted_x_i, dim=0, keepdim=True)
            for attn_weighted_x_i in torch.split(attn_weighted_x, lengths)
        ])
        return context_vectors, attn_weights
