import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import h5py
from pdb import set_trace as st


def weight_normalize(weights):
    return weights ** 2 / torch.sum(weights, dim=0)


class wFMLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_neighbors, num_points, down_sample_rate=1):
        super(wFMLayer, self).__init__()

        # Initialize Weights
        self.w = nn.Parameter(torch.rand(in_channels, num_neighbors, out_channels))
        
        # Configurations
        self.k = num_neighbors
        self.down_sample_rate = down_sample_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Sequential
        self.linear = nn.Sequential(
            nn.Conv2d(num_points, num_points, (25, in_channels)),
            nn.Sigmoid(),
        )


    def wFM_on_sphere(self, input_set, knn_matrix=None):

#         print("---------------------------------\n[wFMLayer]")
#         print("===\nSize: {}".format(self.w.size()))
#         print("===\nWeight: \n{}\n".format(self.w))

        # Get Dimensions of Input
        B, N, D, C = input_set.shape
        v = self.linear(input_set)
        input_set = input_set.contiguous()
        input_set = input_set.view(B, N, D*C)

        # Downsampling
        if self.down_sample_rate != 1:
            input_set = down_sampling(input_set, v.squeeze(), int(N * self.down_sample_rate))
            N = int(N * self.down_sample_rate)

        input_set = input_set.view(B, N, D, C)
        idx = torch.arange(B) * N # IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) # reshape to be added to knn indices

        # Combine in * k and normalize there
        # Get [B * N * K * D * C]
        k2 = knn_matrix + idx
        ptcld = input_set.view(B*N, D, C) # [(B*N) * (D*C)]
        ptcld = ptcld.view(B*N, D*C)
        gathered = ptcld[k2] # [B * N * K * (D*C)]
        gathered = gathered.view(B, N, self.k, D, C) # [B * N * K * D * C]

        gathered = gathered.permute(0, 1, 3, 4, 2) # [B * N * D * C * K]
        
        normalized_w = self.w.view(self.in_channels * self.k, self.out_channels)
        normalized_w = weight_normalize(normalized_w)
        
        gathered = gathered.contiguous()
        gathered = gathered.view(B, N, D, C * self.k) # [B * N * D * C * K]
        gathered = torch.matmul(gathered, normalized_w) # [B * N * D * Cout]

        return gathered

    def forward(self, x, knn_matrix):
        return self.wFM_on_sphere(x, knn_matrix)
    
    
    
    
class Last(nn.Module):
    def __init__(self, in_channels, out_channels, num_points):
        super(Last, self).__init__()
        self.points = num_points
        self.in_channels = in_channels
        self.w = nn.Parameter(torch.rand(in_channels), requires_grad=True)
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels*num_points, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )

    def FM_on_sphere(self, input_set):

#         print("----------------------------------\nLast")
#         print("===\nSize: {}".format(self.w.size()))
#         print("===\nWeight:\n{}\n===".format(self.w))

        B, N, D, C = input_set.shape
        #st()
        weighted_mean = torch.sum(weight_normalize(self.w) * input_set, dim=3, keepdim=True)
        #print(input_set)
        dist = torch.norm(weighted_mean - input_set, p=2, dim=2) # [B * N * C]

        return dist.view(-1, self.points*self.in_channels)#torch.max(dist, dim = 1)[0] # [B * C]

    def forward(self, x):
        return self.linear2(self.FM_on_sphere(x))

    
class Nonlinear(nn.Module):
    def __init__(self):
        super(Nonlinear, self).__init__()
        self.w = nn.Parameter(torch.rand((1)))

    def nonlinear(self, x):

        # print("-------------------------------\nNonLinear")
        # print("===\nSize: {}".format(self.w.size()))
        # print("===\nWeight:\n{}\n===".format(self.w))

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