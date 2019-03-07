import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import h5py
from pdb import set_trace as st


def weightNormalize(weights_in):
    return weights_in**2/ torch.sum(weights_in**2) #torch.stack(out_all)

class wFMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbor, num_points, down_sample=1):
        super(wFMLayer, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'
        #dont forget to normalize w in dim 0
        self.w1 = nn.Parameter(torch.rand(in_channels, num_neighbor))
        self.w2 = nn.Parameter(torch.rand(out_channels, in_channels))
        #self.weights = nn.Parameter(torch.randn(in_channels, num_neighbor, out_channels))
        self.neighbors = num_neighbor
        self.out_channels = out_channels
        self.down_sample = down_sample
#         self.linear = nn.Sequential(
#             nn.Conv2d(num_points, num_points, (25, in_channels)),
#             nn.Sigmoid(),
#         )
        #self.G = GumblerSinkhorn(int(down_sample*num_points), num_points)


    #Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, input_set, adj_mtr=None):
        #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = input_set.shape
        #v = self.linear(input_set)
        input_set = input_set.contiguous()
        input_set = input_set.view(B, N, D*C)
#         if self.down_sample != 1:
#             input_set = down_sampling(input_set, v.squeeze(), int(N*self.down_sample))
#             N = int(N*self.down_sample)
        #print(input_set.shape)
        if adj_mtr is None:
            adj_mtr=pairwise_distance(input_set)
        input_set=input_set.view(B, N, D, C)
        k=self.neighbors #This is number of neighbors
        idx = torch.arange(B)*N #IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) #reshape to be added to knn indices
        
        k2 = knn(adj_mtr, k=k, include_myself=True) #B*N*k
  
        k2 = torch.Tensor(k2).long()+idx
        ptcld = input_set.view(B*N, D, C) #reshape pointset to BN * DC
        ptcld = ptcld.view(B*N, D*C)
        gathered=ptcld[k2] #get matrix of dimension B*N*K*(D*C)
        gathered = gathered.view(B*N, k, D, C)
        gathered = gathered.view(B,N,k,D,C)

        q_p_s = gathered


        #####Project points onto tangent plane on north pole######
#         north_pole_cos = torch.zeros(gathered.shape).cuda()
#         theta = torch.acos(torch.clamp(gathered[:, :, :, 0, :], -1, 1)) #this is of shape B*N*K*C
#         eps = (torch.ones(theta.shape)*0.0001).cuda()
#         theta_sin = theta / (torch.sin(theta) + eps ) #theta/sin(theta) B*N*K*D*C
#         north_pole_cos[:, :, :, 0, :] = torch.cos(theta) #cos(theta)
#         q_p = gathered - north_pole_cos #q-cos(theta)
#         theta_sin = theta_sin.repeat(1, 1, 1, D) #should be of shape B*N*K*D*C
#         theta_sin = theta_sin.view(B, N, k, D, C)
#         q_p_s = torch.mul(q_p, theta_sin) #B*N*K*D*C
        #####End Code######

        q_p_s = q_p_s.permute(0, 1, 3, 4, 2)
        
        m=self.out_channels
        weighted = q_p_s * weightNormalize(self.w1)  
        weighted = torch.sum(weighted, dim = -1) # B*N*D*C
        weighted_sum = torch.matmul(weighted, weightNormalize(self.w2).transpose(1, 0)) 


        #####Project points from tangent plane back to sphere######
#         v_mag = torch.norm(weighted_sum, dim=2)
#         north_pole_cos_vmag = torch.zeros(weighted_sum.shape).cuda()
#         north_pole_cos_vmag[:, :, 0, :] = torch.cos(v_mag)
#         normed_w = F.normalize(weighted_sum, p=2, dim=2)
#         sin_vmag = torch.sin(v_mag).repeat(1, 1, D).view(B, N, D, m)
#         out = north_pole_cos_vmag + sin_vmag*normed_w
        #####End Code#####
        return weighted_sum
    
    ## to do: implement inverse exponential mapping
    def forward(self, x, adj_mtr=None):
        return self.wFM_on_sphere(x, adj_mtr)

class Last(nn.Module):
    def __init__(self):
        super(Last, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'

    #Initial input is B * N * C * d ----> B * N1 * C * m
    def FM_on_sphere(self, input_set):
        #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = input_set.shape

        ####Project points onto tangent plane on north pole#####
#         north_pole_cos = torch.zeros(input_set.shape).cuda() #B*N*D*C
#         theta = torch.acos(torch.clamp(input_set[:, :, 0, :], -1, 1)) #this is of shape B*N*D*C
#         eps = (torch.ones(theta.shape)*0.0001).cuda()
#         theta_sin = theta / (torch.sin(theta) + eps) #theta/sin(theta) B*N*K*D*C
#         north_pole_cos[:, :, 0, :] = torch.cos(theta) #cos(theta)
#         q_p = input_set - north_pole_cos #q-cos(theta)
#         theta_sin = theta_sin.repeat(1, 1, D) #should be of shape B*N*K*D*C
#         theta_sin = theta_sin.view(B, N, D, C)
#         q_p_s = torch.mul(q_p, theta_sin) #B*N*D*C
        #####End Code######
        
        q_p_s = input_set
        unweighted_sum = torch.mean(q_p_s, 3, keepdim= True) #B*N*D*C
        dist = torch.norm(unweighted_sum - q_p_s, p=2, dim=2) #B*N*C
        return torch.max(dist, dim = 1)[0] #B*C
    
    def forward(self, x):
        st()
        return torch.max(x.view(-1,x.shape[1],x.shape[2]*x.shape[3]), dim=1)[0]

def sdt(x, grid = 20, sigma = 1):
    dim = x.shape[2]
    num_point = x.shape[1]
    out = np.zeros((x.shape[0],x.shape[1],grid**dim,1)).cuda()
    linspace = np.linspace(0,1,grid)
    mesh = linspace
    for i in range(dim-1):
        mesh = np.meshgrid(mesh, linspace)
    mesh = np.array(mesh)
    mesh = mesh.reshape(mesh.shape[0], -1)
    for batch_id in range(x.shape[0]):
        for id_, var in enumerate(mesh.T):
            var = var.reshape((1, -1))
            core_dis = np.sum( (np.squeeze(x[batch_id, ...]) -  np.repeat(var, num_point, axis = 0) ) **2, axis =1) *1.0 /(2*sigma)
            out[batch_id, :, id_,0] = np.exp( -core_dis)
    return out


# class GumblerSinkhorn(nn.Module):
#     def __init__(self, dim_decrease, original_dim, T=0.1):
#         super(GumblerSinkhorn, self).__init__()
# #         self.u = nn.Parameter(torch.randperm(dim_decrease).float().unsqueeze(-1))
# #         self.v = nn.Parameter(torch.randperm(original_dim).float().unsqueeze(-1))
# #         self.u = nn.Parameter(torch.rand(dim_decrease, 1))
# #         self.v = nn.Parameter(torch.rand(original_dim, 1))
#         self.w = nn.Parameter(torch.rand(dim_decrease, original_dim))
#         self.T = T
    
#     def transform(self, input_set, times=20):
#         e_weight = self.w
#         e_weight = torch.exp(e_weight/self.T)
#         for i in range(times):
#             e_weight = e_weight / torch.sum(e_weight, dim=1, keepdim = True)
#             e_weight = e_weight / torch.sum(e_weight, dim=0, keepdim = True)
#         return torch.matmul(e_weight, input_set)

#     def forward(self, inputs):
#         return self.transform(inputs)
