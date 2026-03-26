# PyTorch
import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', neighbors=1):
        super().__init__()

        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        assert pooling in ['Max', 'Mean', 'CenterGaussian', 'ABMIL', 'SmAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'CenterGaussian':
            self.pool = layers.CenterGaussian()
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

        assert pooling in ['Max', 'Mean', 'CenterGaussian', 'ABMIL', 'TransMIL', 'SmAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'CenterGaussian':
            self.pool = layers.CenterGaussian()
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

class InstanceClassifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        out = attn_logits = self.clf(x)
        attn_weights = torch.cat([
            torch.nn.functional.softmax(attn_logits_i, dim=0)
            for attn_logits_i in torch.split(attn_logits, lengths)
        ])
        return out, attn_weights
