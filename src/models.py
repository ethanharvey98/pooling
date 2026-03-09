import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', neighbors=1):
        super().__init__()

        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        assert pooling in ['Max', 'Mean', 'ABMIL', 'SmAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=out_features)
        elif pooling == 'SmAP':
            self.pool = layers.SmAP(in_features=out_features, neighbors=neighbors)

    def forward(self, x, lengths):
        out = self.clf(x)                                
        out, attn_weights = self.pool(out, lengths)
        return out, attn_weights
    
class PoolClf(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', num_heads=8, neighbors=1):
        super().__init__()

        assert pooling in ['Max', 'Mean', 'ABMIL', 'TransMIL', 'SmAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=in_features)
        elif pooling == 'TransMIL':
            self.pool = layers.TransMIL(in_features=in_features, num_heads=num_heads)
        elif pooling == 'SmAP':
            self.pool = layers.SmAP(in_features=in_features, neighbors=neighbors)
            
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        out, attn_weights = self.pool(x, lengths)
        out = self.clf(out)
        return out, attn_weights
    
class VAPGaussianMIL(torch.nn.Module):
    def __init__(self, in_features, out_features=1, hidden_dim=128):
        super().__init__()
        self.pool = layers.VAPGaussianAttention(in_features, hidden_dim)
        self.clf = torch.nn.Linear(in_features, out_features)

    def forward(self, x, lengths):
        # Default: sample during training, deterministic during eval
        # predict_with_uncertainty overrides this explicitly
        if not hasattr(self, '_uncertainty_mode'):
            self.pool.deterministic = not self.training
        out, attn_weights = self.pool(x, lengths)
        logits = self.clf(out)
        self.kl_loss = self.pool.kl_loss
        return logits, attn_weights

    def get_attention_weights(self):
        return {
            'mu': self.pool._mu,
            'log_sigma': self.pool._log_sigma,
            'attn_weights': self.pool._last_attn_weights,
        }

    def predict_with_uncertainty(self, x, lengths, n_samples=10):
        self._uncertainty_mode = True
        self.pool.deterministic = False
        logits_list, attn_list = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                logits, attn = self.forward(x, lengths)
                logits_list.append(logits)
                attn_list.append(attn)
        self.pool.deterministic = True
        del self._uncertainty_mode
        return {
            'logits_mean': torch.stack(logits_list).mean(0),
            'logits_var': torch.stack(logits_list).var(0),
            'attn_samples': torch.stack(attn_list),
        }


class VAPBernoulliMIL(torch.nn.Module):
    def __init__(self, in_features, out_features=1, hidden_dim=128,
                 estimator='gumbel', pi_0=0.1, tau_start=1.0, tau_min=0.1, anneal_rate=0.95):
        super().__init__()
        self.pool = layers.VAPBernoulliAttention(
            in_features, hidden_dim, estimator=estimator,
            pi_0=pi_0, tau_start=tau_start, tau_min=tau_min, anneal_rate=anneal_rate,
        )
        self.clf = torch.nn.Linear(in_features, out_features)

    def forward(self, x, lengths):
        out, attn_weights = self.pool(x, lengths)
        logits = self.clf(out)
        self.kl_loss = self.pool.kl_loss
        return logits, attn_weights

    def get_attention_weights(self):
        return {
            'probs': self.pool._probs,
            'z_mask': self.pool._z_mask,
        }

    def predict_with_uncertainty(self, x, lengths, n_samples=10):
        was_training = self.training
        self.train()
        logits_list, attn_list = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                logits, attn = self.forward(x, lengths)
                logits_list.append(logits)
                attn_list.append(attn)
        if not was_training:
            self.eval()
        return {
            'logits_mean': torch.stack(logits_list).mean(0),
            'logits_var': torch.stack(logits_list).var(0),
            'attn_samples': torch.stack(attn_list),
        }


class VAPGaussianSparseMIL(torch.nn.Module):
    def __init__(self, in_features, out_features=1, hidden_dim=128, prior_scale=1.0):
        super().__init__()
        self.pool = layers.VAPGaussianSparseAttention(in_features, hidden_dim, prior_scale=prior_scale)
        self.clf = torch.nn.Linear(in_features, out_features)

    def forward(self, x, lengths):
        if not hasattr(self, '_uncertainty_mode'):
            self.pool.deterministic = not self.training
        out, attn_weights = self.pool(x, lengths)
        logits = self.clf(out)
        self.kl_loss = self.pool.kl_loss
        return logits, attn_weights

    def get_attention_weights(self):
        return {
            'mu': self.pool._mu,
            'log_sigma': self.pool._log_sigma,
            'attn_weights': self.pool._last_attn_weights,
        }

    def predict_with_uncertainty(self, x, lengths, n_samples=10):
        self._uncertainty_mode = True
        self.pool.deterministic = False
        logits_list, attn_list = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                logits, attn = self.forward(x, lengths)
                logits_list.append(logits)
                attn_list.append(attn)
        self.pool.deterministic = True
        del self._uncertainty_mode
        return {
            'logits_mean': torch.stack(logits_list).mean(0),
            'logits_var': torch.stack(logits_list).var(0),
            'attn_samples': torch.stack(attn_list),
        }


class OnTheDesign(torch.nn.Module):
    def __init__(self, num_classes, expansion=4, type_name="conv3x3x3", norm_type="Instance"):
        super().__init__()
        self.num_classes = num_classes
        self.conv = torch.nn.Sequential()

        self.conv.add_module("conv0_s1", torch.nn.Conv3d(in_channels=2, out_channels=4*expansion, kernel_size=1))

        if norm_type == "Instance":
            self.conv.add_module("lrn0_s1", torch.nn.InstanceNorm3d(num_features=4*expansion))
        else:
            self.conv.add_module("lrn0_s1", torch.nn.BatchNorm3d(num_features=4*expansion))
        self.conv.add_module("relu0_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool0_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv1_s1", torch.nn.Conv3d(in_channels=4*expansion, out_channels=32*expansion, kernel_size=3, padding=0, dilation=2))
        
        if norm_type == "Instance":
            self.conv.add_module("lrn1_s1", torch.nn.InstanceNorm3d(num_features=32*expansion))
        else:
            self.conv.add_module("lrn1_s1", torch.nn.BatchNorm3d(num_features=32*expansion))
        self.conv.add_module("relu1_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool1_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv2_s1", torch.nn.Conv3d(in_channels=32*expansion, out_channels=64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == "Instance":
            self.conv.add_module("lrn2_s1", torch.nn.InstanceNorm3d(num_features=64*expansion))
        else:
            self.conv.add_module("lrn2_s1", torch.nn.BatchNorm3d(num_features=64*expansion))
        self.conv.add_module("relu2_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool2_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv3_s1", torch.nn.Conv3d(in_channels=64*expansion, out_channels=64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == "Instance":
            self.conv.add_module("lrn3_s1", torch.nn.InstanceNorm3d(num_features=64*expansion))
        else:
            self.conv.add_module("lrn2_s1", torch.nn.BatchNorm3d(num_features=64*expansion))
        self.conv.add_module("relu3_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool2_s1", torch.nn.MaxPool3d(kernel_size=5, stride=2))

        self.head = torch.nn.Sequential()
        self.head.add_module("fc0_s1", torch.nn.Linear(in_features=15*21*21*64*expansion, out_features=1024))
        self.head.add_module("relu4_s1", torch.nn.ReLU(inplace=True))
        self.head.add_module("fc1_s1", torch.nn.Linear(in_features=1024, out_features=self.num_classes))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.head(x.flatten(start_dim=1, end_dim=-1))
        return x
