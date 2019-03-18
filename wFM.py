import torch
import time
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import h5py
from pdb import set_trace as st
import utils

def weight_normalize(weights, dim):
    w_sqrd = weights**2
    return w_sqrd / torch.sum(w_sqrd, dim=dim, keepdim=True)


class wFMLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_neighbors, num_points, grid_size, down_sample_rate=1):
        super(wFMLayer, self).__init__()

        # Initialize Weights
        self.w1 = nn.Parameter(torch.rand(in_channels, num_neighbors))
        self.w2 = nn.Parameter(torch.rand(in_channels, out_channels))
        
        # Configurations
        self.k = num_neighbors
        self.down_sample_rate = down_sample_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Sequential
        self.conv = nn.Sequential(
            nn.Conv2d(num_points, num_points, (grid_size**2, in_channels)),
            nn.Sigmoid(),
        )


    def wFM_on_sphere(self, inputs):

#         print("---------------------------------\n[wFMLayer]")
#         print("===\nSize: {}".format(self.w.size()))
#         print("===\nWeight: \n{}\n".format(self.w))
#         print("---------------------------------\n")

        # Get Dimensions of Input
        B, N, D, C = inputs.shape
        v = self.conv(inputs)
        inputs = inputs.contiguous()
        inputs = inputs.view(B, N, D*C)

        # Downsampling
        if self.down_sample_rate != 1:
            inputs = down_sampling(inputs, v.squeeze(), int(N * self.down_sample_rate))
            N = int(N * self.down_sample_rate)
        inputs = inputs.view(B, N, D, C)

        # Get KNN Matrix
        adj = utils.pairwise_distance(inputs)
        
        print("---------------------------------\n[Adj Matrix")
        print(adj)
        print("---------------------------------\n")
        
        knn_matrix = utils.knn(adj, k=self.k, include_myself=True)
        knn_matrix = torch.Tensor(knn_matrix).long()
        
        idx = torch.arange(B) * N # IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) # reshape to be added to knn indices

        # Combine in * k and normalize there
        # Get [B * N * K * D * C]
        k2 = knn_matrix + idx
        ptcld = inputs.view(B*N, D, C) # [(B*N) * (D*C)]
        ptcld = ptcld.view(B*N, D*C)
        gathered = ptcld[k2] # [B * N * K * (D*C)]
        gathered = gathered.view(B, N, self.k, D, C) # [B * N * K * D * C]

        gathered = gathered.permute(0, 1, 3, 4, 2) # [B * N * D * C * K]
        
        weighted = gathered * weight_normalize(self.w1, dim=1) # [B * N * D * C * K]
        weighted = torch.sum(weighted, dim=-1) # [B * N * D * C]
        weighted = torch.matmul(weighted, weight_normalize(self.w2, dim=0)) # [B * N * D * Cout]
        
        return weighted

    def forward(self, inputs):
        return self.wFM_on_sphere(inputs)
    
    
class Last(nn.Module):
    def __init__(self, in_channels, out_channels, num_points):
        super(Last, self).__init__()
        self.points = num_points
        self.in_channels = in_channels
        #self.w1 = nn.Parameter(torch.rand(num_points, in_channels))
        self.w1 = nn.Parameter(torch.rand(in_channels))
        self.w2 = nn.Parameter(torch.rand(num_points,10))
        
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def FM_on_sphere(self, inputs):

#         print("----------------------------------\n[Last]")
#         print("===\nSize: {}".format(self.w.size()))
#         print("===\nWeight:\n{}\n===".format(self.w))
#         print("----------------------------------\n")

        B, N, D, C = inputs.shape # [B * N * D * C]
        
        channel_mean = torch.sum(weight_normalize(self.w1, dim=0) * inputs, dim=3, keepdim=True) # [B * N * D * 1]
 
        channel_diff = inputs - channel_mean # [B * N * D * C]
        dim_diff = channel_diff.transpose(2, 3) # [B * N * C * D]
        
        dist = torch.norm(dim_diff, dim=3).transpose(1,2) # [B * C * N]
        dist = torch.matmul(dist, self.w2) # [B * C * 10]

        return torch.max(dist, dim=2)[0]

    def forward(self, inputs):
        
        inputs = self.FM_on_sphere(inputs) # [B * C]
        return self.linear2(inputs) # [B * Cout]

    
class Nonlinear(nn.Module):
    def __init__(self):
        super(Nonlinear, self).__init__()
        self.w = nn.Parameter(torch.rand((1)))

    def nonlinear(self, x):

#         print("---------------------------------\n[NonLinear]")
#         print("===\nSize: {}".format(self.w.size()))
#         print("===\nWeight:\n{}\n===".format(self.w))
#         print("---------------------------------\n")

        B, N, D, C = x.shape
        weights = torch.sigmoid(self.w)
        weights = weights.repeat(C)
        n_weights_plus_one = D * weights + torch.ones(weights.shape).cuda()
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        n_weights_plus_one = n_weights_plus_one.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (x + weights) / n_weights_plus_one

    def forward(self, x):
        return self.nonlinear(x)


























# END