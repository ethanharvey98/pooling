# PyTorch
import torch
# Importing our custom module(s)
import utils

class MILDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, lengths, y):
        super().__init__()
        self.X = X
        self.X_split = torch.split(X, lengths)
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, index):
        return self.X_split[index], self.lengths[index], self.y[index]

class ShiftedMeanMILDataset(torch.utils.data.Dataset):
    
    def __init__(self, N, R=3, S_low=15, S_high=45, K=1, M=768, p_y1=0.5, Delta=1.0, mu=0.0, sigma=1.0, seed=42):
        super().__init__()
        
        self.N = N
        self.R = R
        self.S_low = S_low
        self.S_high = S_high
        self.K = K
        self.M = M
        self.p_y1 = p_y1
        self.Delta = Delta
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        
        self.generator = torch.Generator().manual_seed(self.seed)
        self.lengths = tuple(torch.randint(self.S_low, self.S_high + 1, (self.N,), generator=self.generator).tolist())
        self.H = self.mu + self.sigma * torch.randn(sum(self.lengths), self.M, generator=self.generator)
        self.u = torch.cat([torch.randint(0, S_i - self.R + 1, (1,), generator=self.generator) for S_i in self.lengths])
        self.y = torch.bernoulli(self.p_y1 * torch.ones(size=(self.N,)), generator=self.generator).int().reshape(-1, 1).float()
        self.H_split = torch.split(self.H, self.lengths)
        
        for i, H_i in enumerate(self.H_split):
            if self.y[i] == 1:
                H_i[self.u[i]:self.u[i] + self.R, 0:self.K] += self.Delta
                
    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return self.H_split[index], self.lengths[index], self.y[index]

    def p_y1_given_h(self, index):
        h = self.H_split[index][:, 0:self.K]
        S_i = self.lengths[index]
        p_h_given_y0 = torch.prod(torch.stack([utils.normal_pdf(h[j]) for j in range(S_i)])) * (1.0 - self.p_y1)
        p_u = (1 / (S_i - self.R + 1)) * torch.ones(size=(S_i - self.R + 1,))
        # p_h_given_u_y1.shape = (S_i - R + 1, S_i, K)
        p_h_given_u_y1 = torch.stack([torch.stack([torch.stack([utils.normal_pdf(h[j, k], self.mu + self.Delta) if j >= u and j < (u + self.R) else utils.normal_pdf(h[j, k]) for k in range(self.K)]) for j in range(S_i)]) for u in range(S_i - self.R + 1)])
        p_h_given_y1 = torch.sum(torch.prod(torch.prod(p_h_given_u_y1, dim=-1), dim=-1) * p_u, dim=-1) * self.p_y1
        p_y1_given_h = p_h_given_y1 / (p_h_given_y0 + p_h_given_y1)
        return p_y1_given_h
