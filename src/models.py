# PyTorch
import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max'):
        super().__init__()
                            
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
                                    
        assert pooling in ['Max', 'Mean', 'ABMIL', 'smAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=out_features)
        elif pooling == 'smAP':
            self.pool = layers.smAP(in_features=out_features)

    def forward(self, x, lengths):
        out = self.clf(x)                                
        out, attn_weights = self.pool(out, lengths)
        return out, attn_weights
    
class PoolClf(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', num_heads=8):
        super().__init__()
                                                        
        assert pooling in ['Max', 'Mean', 'ABMIL', 'TransMIL', 'smAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=in_features)
        elif pooling == 'TransMIL':
            self.pool = layers.TransMIL(in_features=in_features, num_heads=num_heads)
        elif pooling == 'smAP':
            self.pool = layers.smAP(in_features=in_features)
            
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        out, attn_weights = self.pool(x, lengths)
        out = self.clf(out)
        return out, attn_weights
    