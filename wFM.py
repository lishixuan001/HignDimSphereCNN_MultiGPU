import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import h5py
from pdb import set_trace as st


def weightNormalize(weights_in):
    return weights_in ** 2 / torch.sum(weights_in ** 2)

class wFMLayer1(nn.Module):
    def __init__(self, in_channels, num_neighbors, num_points, down_sample_rate=1):
        super(wFMLayer1, self).__init__()

        # Initialize Weights
        self.w1 = nn.Parameter(torch.rand(in_channels, num_neighbors))

        # Configurations
        self.k = num_neighbors
        self.down_sample_rate = down_sample_rate

        # Sequential
        self.linear = nn.Sequential(
            nn.Conv2d(num_points, num_points, (25, in_channels)),
            nn.Sigmoid(),
        )


    def wFM_on_sphere(self, input_set, knn_matrix=None):

        # print("---------------------------------\n[wFMLayer1]")
        # print("===\nSize: {}".format(self.w1.size()))
        # print("===\nWeight 1: \n{}\n".format(self.w1))
        print(input_set.shape)
        # Get Dimensions of Input
        B, N, D, C = input_set.shape
        v = self.linear(input_set)
        input_set = input_set.view(B, N, D*C)

        # Downsampling
        if self.down_sample_rate != 1:
            input_set = down_sampling(input_set, v.squeeze(), int(N * self.down_sample_rate))
            N = int(N * self.down_sample_rate)

        input_set = input_set.view(B, N, D, C)
        idx = torch.arange(B) * N # IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) # reshape to be added to knn indices

        # Get [B * N * K * D * C]
        k2 = knn_matrix + idx
        ptcld = input_set.view(B*N, D, C) # [(B*N) * (D*C)]
        ptcld = ptcld.view(B*N, D*C)
        gathered = ptcld[k2] # [B * N * K * (D*C)]
        gathered = gathered.view(B, N, self.k, D, C)

        gathered = gathered.permute(0, 1, 3, 4, 2) # [B * N * D * C * K]

        # Get Weighted Results
        weights_sqrd = self.w1 ** 2
        weights_check = weights_sqrd / torch.sum(weights_sqrd, dim=1, keepdim=True)
        
        weighted = gathered * weights_check
        weighted = torch.sum(weighted, dim=-1) # [B * N * D * C]

        return weighted

    def forward(self, x, knn_matrix):
        return self.wFM_on_sphere(x, knn_matrix)



class wFMLayer2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(wFMLayer2, self).__init__()

        # Initial Weights
        self.w2 = nn.Parameter(torch.rand(in_channels, out_channels))

    def transition_channels(self, weighted):

        # print("---------------------------------\n[wFMLayer2]")
        # print("===\nSize: {}".format(self.w2.size()))
        # print("===\nWeight 2: \n{}\n".format(self.w2))
        #weighted_check = weightNormalize(self.w2)
        weights_sqrd = self.w2**2
        weighted_check = weights_sqrd / torch.sum(weights_sqrd, dim=0, keepdim=True)
            
        weighted_sum = torch.matmul(weighted, weighted_check)
        return weighted_sum

    def forward(self, weighted):
        return self.transition_channels(weighted)



class wFMLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_neighbors, num_points, down_sample_rate=1):
        super(wFMLayer, self).__init__()

        # Initialize Weights
        self.w = nn.Parameter(torch.rand(in_channels, num_neighbors, out_channels))

        # Configurations
        self.k = num_neighbors
        self.down_sample_rate = down_sample_rate
        self.ins = in_channels
        self.outs = out_channels
        # Sequential
        self.linear = nn.Sequential(
            nn.Conv2d(num_points, num_points, (25, in_channels)),
            nn.Sigmoid(),
        )


    def wFM_on_sphere(self, input_set, knn_matrix=None):

        # print("---------------------------------\n[wFMLayer1]")
        # print("===\nSize: {}".format(self.w1.size()))
        # print("===\nWeight 1: \n{}\n".format(self.w1))
        print(input_set.shape)
        # Get Dimensions of Input
        B, N, D, C = input_set.shape
        v = self.linear(input_set)
        input_set = input_set.view(B, N, D*C)

        # Downsampling
        if self.down_sample_rate != 1:
            input_set = down_sampling(input_set, v.squeeze(), int(N * self.down_sample_rate))
            N = int(N * self.down_sample_rate)

        input_set = input_set.view(B, N, D, C)
        idx = torch.arange(B) * N # IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) # reshape to be added to knn indices

        #combine in * k and normalize there
        # Get [B * N * K * D * C]
        k2 = knn_matrix + idx
        ptcld = input_set.view(B*N, D, C) # [(B*N) * (D*C)]
        ptcld = ptcld.view(B*N, D*C)
        gathered = ptcld[k2] # [B * N * K * (D*C)]
        gathered = gathered.view(B, N, self.k, D, C)

        gathered = gathered.permute(0, 1, 3, 4, 2) # [B * N * D * C * K]
        
        normalized_w = self.w.view(self.ins*self.k, self.outs) ** 2
        normalized_w = (normalized_w / torch.sum(normalized_w, dim=0))
        gathered = gathered.contiguous()
        gathered = gathered.view(B, N, D, C*self.k)
        gathered = torch.matmul(gathered, normalized_w)
        # Get Weighted Results # [B * N * D * C]

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

        # print("----------------------------------\nLast")
        # print("===\nSize: {}".format(self.w.size()))
        # print("===\nWeight:\n{}\n===".format(self.w))

        B, N, D, C = input_set.shape
        #st()
        weighted_mean = torch.sum(weightNormalize(self.w) * input_set, dim=3, keepdim=True)
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