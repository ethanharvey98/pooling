import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', neighbors=1, **pool_kwargs):
        super().__init__()

        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        assert pooling in ['Max', 'Mean', 'ABMIL', 'ABMILMaxInst', 'SmAP', 'BernoulliVAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=out_features)
        elif pooling == 'ABMILMaxInst':
            self.pool = layers.ABMILMaxInst(in_features=out_features)
        elif pooling == 'SmAP':
            self.pool = layers.SmAP(in_features=out_features, neighbors=neighbors)
        elif pooling == 'BernoulliVAP':
            self.pool = layers.BernoulliVAP(in_features=out_features, **pool_kwargs)

    def forward(self, x, lengths):
        out = self.clf(x)                                
        out, attn_weights = self.pool(out, lengths)
        return out, attn_weights
    
class PoolClf(torch.nn.Module):
    def __init__(self, in_features, out_features, pooling='Max', num_heads=8, neighbors=1, **pool_kwargs):
        super().__init__()

        assert pooling in ['Max', 'Mean', 'ABMIL', 'ABMILMaxInst', 'TransMIL', 'SmAP', 'BernoulliVAP']
        if pooling == 'Max':
            self.pool = layers.Max()
        elif pooling == 'Mean':
            self.pool = layers.Mean()
        elif pooling == 'ABMIL':
            self.pool = layers.ABMIL(in_features=in_features)
        elif pooling == 'ABMILMaxInst':
            self.pool = layers.ABMILMaxInst(in_features=in_features)
        elif pooling == 'TransMIL':
            self.pool = layers.TransMIL(in_features=in_features, num_heads=num_heads)
        elif pooling == 'SmAP':
            self.pool = layers.SmAP(in_features=in_features, neighbors=neighbors)
        elif pooling == 'BernoulliVAP':
            self.pool = layers.BernoulliVAP(in_features=in_features, **pool_kwargs)
            
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        out, attn_weights = self.pool(x, lengths)
        out = self.clf(out)
        return out, attn_weights
    
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
