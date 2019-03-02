import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import h5py
from pdb import set_trace as st

def weightNormalize(weights_in):
    weights = weights_in**2
    weights = weights/ torch.sum(weights, dim = 1, keepdim= True)
    return weights #torch.stack(out_all)

class wFMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbor):
        super(wFMLayer, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'
        #dont forget to normalize w in dim 0
        self.w1 = nn.Parameter(torch.randn(in_channels, num_neighbor))
        self.w2 = nn.Parameter(torch.randn(out_channels, in_channels))
        #self.weights = nn.Parameter(torch.randn(in_channels, num_neighbor, out_channels))
        self.neighbors = num_neighbor
        self.out_channels = out_channels


    #Initial input is B * N * C * d ----> B * N1 * C * m
    def wFM_on_sphere(self, input_set, adj_mtr=None):
        #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = input_set.shape
        k=self.neighbors #Tis is number of neighbors
        idx = torch.arange(B)*N #IDs for later processing, used because we flatten the tensor
        idx = idx.view((B, 1, 1)) #reshape to be added to knn indices
        if (adj_mtr is not None):
            adj_mtr=pairwise_distance(input_set)
        k2 = knn(adj_mtr, k=k, include_myself=True) #B*N*k
        k2 = torch.Tensor(k2).long()+idx
        ptcld = input_set.view(B*N, D, C) #reshape pointset to BN * DC
        ptcld = ptcld.view(B*N, D*C)
        gathered=ptcld[k2] #get matrix of dimension B*N*K*(D*C)
        gathered = gathered.view(B, N, k, D, C)

        q_p_s = gathered


        ######Project points onto tangent plane on north pole######
        # north_pole_cos = torch.zeros(gathered.shape).cuda()
        # theta = torch.acos(torch.clamp(gathered[:, :, :, 0, :], -1, 1)) #this is of shape B*N*K*C
        # eps = (torch.ones(theta.shape)*0.0001).cuda()
        # theta_sin = theta / (torch.sin(theta) + eps ) #theta/sin(theta) B*N*K*D*C
        # north_pole_cos[:, :, :, 0, :] = torch.cos(theta) #cos(theta)
        # q_p = gathered - north_pole_cos #q-cos(theta)
        # theta_sin = theta_sin.repeat(1, 1, 1, D) #should be of shape B*N*K*D*C
        # theta_sin = theta_sin.view(B, N, k, D, C)
        # q_p_s = torch.mul(q_p, theta_sin) #B*N*K*D*C
        ######End Code######

        q_p_s = torch.transpose(q_p_s, 2, 3)
        q_p_s = torch.transpose(q_p_s, 3, 4) #Reshape to B*N*D*C*k
        transformed_w1 = weightNormalize(self.w1)
        transformed_w2 = weightNormalize(self.w2).transpose(1, 0)
        m=self.out_channels
        weighted = q_p_s * transformed_w1
        weighted = torch.sum(weighted, dim = -1)
        st()
        weighted_sum = torch.bmm(weighted, transformed_w2)

        ######Project points from tangent plane back to sphere######
        # v_mag = torch.norm(weighted_sum, dim=2)
        # north_pole_cos_vmag = torch.zeros(weighted_sum.shape).cuda()
        # north_pole_cos_vmag[:, :, 0, :] = torch.cos(v_mag)
        # normed_w = F.normalize(weighted_sum, p=2, dim=2)
        # sin_vmag = torch.sin(v_mag).repeat(1, 1, D).view(B, N, D, m)
        # out = north_pole_cos_vmag + sin_vmag*normed_w
        ######End Code#####
        return weighted_sum

    ## to do: implement inverse exponential mapping
    def forward(self, x, adj_mtr=None):
        return self.wFM_on_sphere(x, adj_mtr)

class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        #Initial input is B * N * D * C ----> B * N1 * D * C'
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )


    #Initial input is B * N * C * d ----> B * N1 * C * m
    def FM_on_sphere(self, input_set):
        #Input is B*N*D*C where B is batch size, N is number of points, D is dimension of each point, and C is input channel
        B, N, D, C = input_set.shape

        # #####Project points onto tangent plane on north pole#####
        # north_pole_cos = torch.zeros(input_set.shape).cuda() #B*N*D*C
        # theta = torch.acos(torch.clamp(input_set[:, :, 0, :], -1, 1)) #this is of shape B*N*D*C
        # eps = (torch.ones(theta.shape)*0.0001).cuda()
        # theta_sin = theta / (torch.sin(theta) + eps) #theta/sin(theta) B*N*K*D*C
        # north_pole_cos[:, :, 0, :] = torch.cos(theta) #cos(theta)
        # q_p = input_set - north_pole_cos #q-cos(theta)
        # theta_sin = theta_sin.repeat(1, 1, D) #should be of shape B*N*K*D*C
        # theta_sin = theta_sin.view(B, N, D, C)
        # q_p_s = torch.mul(q_p, theta_sin) #B*N*D*C
        # ######End Code######



        unweighted_sum = torch.mean(q_p_s, 3, keepdim= True) #B*N*D*C
        dist = torch.norm(unweighted_sum - q_p_s, p=2, dim=2) #B*N*C

        return torch.max(dist, dim = 1)[0] #B*C

        '''#distance in terms of cosine
        #for each channel compute distance from mean to get B*N*C reshape to -> B*NC (can also do global maxpool)
        #print(1 in torch.isnan(unweighted_sum).numpy())

        v_mag = torch.norm(unweighted_sum, dim=2)
        north_pole_cos_vmag = torch.zeros(unweighted_sum.shape).cuda()
        north_pole_cos_vmag[:, :, 0] = torch.cos(v_mag)
        normed_w = F.normalize(unweighted_sum, p=2, dim=2)
        sin_vmag = torch.sin(v_mag).repeat(1, D).view(B, N, D)
        out = north_pole_cos_vmag + sin_vmag*normed_w

        out = out.unsqueeze(-1)
        x_ = torch.transpose(input_set, 2, 3)
        # print(input_set.shape)
        res = torch.matmul(x_, out).squeeze(-1)
        #print(res.shape)
        res = torch.acos(torch.clamp(res, -1, 1))
        #print("last layer "+str(1 in torch.isnan(res).numpy()))
        return torch.mean(res, dim = 1) #res.view(B, N*C)'''

    ## to do: implement inverse exponential mapping
    def forward(self, x):
        # print(self.wFM_on_sphere(x))
        return self.linear(self.FM_on_sphere(x))

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
