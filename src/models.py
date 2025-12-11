# PyTorch
import torch
# Importing our custom module(s)
import layers

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, instance_conv=False, kernel_size=3, pooling="max", use_pos_embedding=False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
                
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding:
            self.pos_embedding = layers.PositionalEmbeddingLayer()
            
        self.hidden_dim = self.in_features + 1 if self.use_pos_embedding else self.in_features
        
        self.clf = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.out_features, bias=True)
        
        self.instance_conv = instance_conv
        self.kernel_size = kernel_size
        if self.instance_conv:
            #self.conv = layers.InstanceConv1d(in_features=self.out_features, kernel_size=self.kernel_size)
            self.conv = layers.InstanceConv1d(in_features=self.hidden_dim, kernel_size=self.kernel_size)
                            
        self.pooling = pooling
        if self.pooling == "max":
            self.pool = layers.MaxPooling()
        elif self.pooling == "mean":
            self.pool = layers.MeanPooling()
        elif self.pooling == "attention":
            self.pool = layers.AttentionBasedPooling(in_features=self.out_features)
        elif self.pooling == "self_attention":
            self.pool = layers.SelfAttentionPooling(in_features=self.out_features)
        elif self.pooling == "normal":
            self.pool = layers.NormalPooling(in_features=self.out_features)
        elif self.pooling == "smmil":
            self.pool = layers.SmMILPooling(in_features=self.out_features)
        else:
            raise NotImplementedError(f"The specified pooling operation \"{self.pooling}\" is not implemented.")


    def forward(self, x, lengths):
        
        if self.use_pos_embedding:
            x = self.pos_embedding(x, lengths)
        
        if self.instance_conv:
            x = self.conv(x, lengths)
        
        x = self.clf(x)
        
        #if self.instance_conv:
        #    x = self.conv(x, lengths)
                                
        x, attention_weights = self.pool(x, lengths)
        
        return x, attention_weights
    
class PoolClf(torch.nn.Module):
    def __init__(self, in_features, out_features, instance_conv=False, kernel_size=3, pooling="max", num_heads=8, use_pos_embedding=False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
                
        self.use_pos_embedding = use_pos_embedding
        if self.use_pos_embedding:
            self.pos_embedding = layers.PositionalEmbeddingLayer()
            
        self.hidden_dim = self.in_features + 1 if self.use_pos_embedding else self.in_features
        
        self.instance_conv = instance_conv
        self.kernel_size = kernel_size
        if self.instance_conv:
            #self.conv = layers.InstanceConv1d(in_features=self.out_features, kernel_size=self.kernel_size)
            self.conv = layers.InstanceConv1d(in_features=self.hidden_dim, kernel_size=self.kernel_size)
                                    
        self.pooling = pooling
        if self.pooling == "max":
            self.pool = layers.MaxPooling()
        elif self.pooling == "mean":
            self.pool = layers.MeanPooling()
        elif self.pooling == "attention":
            self.pool = layers.AttentionBasedPooling(in_features=self.hidden_dim)
        elif self.pooling == "transformer":
            self.num_heads = num_heads
            self.pool = layers.TransformerBasedPooling(in_features=self.hidden_dim, num_heads=self.num_heads)
        elif self.pooling == "smmil":
            self.pool = layers.SmMILPooling(in_features=self.hidden_dim)
        else:
            raise NotImplementedError(f"The specified pooling operation \"{self.pooling}\" is not implemented.")

        self.clf = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.out_features, bias=True)


    def forward(self, x, lengths):
        
        if self.use_pos_embedding:
            x = self.pos_embedding(x, lengths)
            
        if self.instance_conv:
            x = self.conv(x, lengths)
                                                
        x, attention_weights = self.pool(x, lengths)
        
        x = self.clf(x)
        
        return x, attention_weights
    