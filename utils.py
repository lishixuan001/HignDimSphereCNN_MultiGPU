import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as func
import argparse
import math

def load_args():
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path',     default='../mnistPC', type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',    default=5,           type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--num_epochs',    default=200,          type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--num_points',    default=512,          type=int,   metavar='N',   help='Number of points in a image')
    parser.add_argument('--log_interval',  default=10,           type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--grid',          default=10,           type=int,   metavar='N',   help='grid of sdt')
    parser.add_argument('--sigma',         default=0.05,         type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--log_dir',       default="./log_dir",  type=str,   metavar='N',   help='directory for logging')
    parser.add_argument('--baselr',        default=0.05 ,        type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--gpu',           default='0,1',        type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--num_neighbors', default=10,           type=int,   metavar='XXX', help='Number of Neighbors')

    args = parser.parse_args()
    return args


def data_generation(inputs, grid_size, sigma):
    B, N, D = inputs.size() # [B, N, 2]
    
    # Raw Data Normalization
    inputs = inputs.transpose(1, 2) # [B, 2, N]
    max_val, _ = torch.max(inputs, dim=2, keepdim=True)
    min_val, _ = torch.min(inputs, dim=2, keepdim=True) # [B, 2]
    inputs = (max_val - inputs) / (max_val - min_val)
    inputs = inputs.transpose(1, 2) # [B, N, 2]
    
    # Generate Grid -- [Hard Code, D=2]
    linspace = torch.linspace(0, 1, steps=grid_size)
    xv, yv = torch.meshgrid(linspace, linspace) # [G, G], [G, G]
    xv, yv = xv.unsqueeze(-1), yv.unsqueeze(-1)
    grid = torch.cat((xv, yv), dim=-1) # [G, G, 2]
    grid = grid.view(-1, 2).cuda() # [G^2, 2]
    
    # Mapping & Normalization
    inputs = inputs.unsqueeze(2).repeat(1, 1, grid_size**2, 1) # [B, N, G^2, 2]
    inputs = inputs - grid
    inputs = torch.norm(inputs, p=2, dim=-1) # [B, N, G^2]
    inputs = torch.div(inputs, -2*np.power(sigma, 2)+1e-10)
    inputs = torch.exp(inputs) # [B, N, G^2]
    inputs = func.normalize(inputs, p=2, dim=2) # [B, N, G^2]
    
    # Channel
    inputs = inputs.unsqueeze(-1)  # [B, N, G^2, 1]
    
    return inputs 
    

def permuteBN(fm):
    fm = fm.permute(0,2,3,1)
    bn = nn.BatchNorm2d(fm.shape[1])
    return fm.permute(0,3,1,2)

def mesh_mat(grid, dim = 2):
    linspace = np.linspace(-1,1,grid)
    if dim ==2:
        mesh = np.meshgrid(linspace, linspace)
    if dim == 6:
        mesh = np.meshgrid(linspace, linspace, linspace, linspace, linspace, linspace)
    mesh = torch.from_numpy(np.array(mesh)).cuda()
    mesh = mesh.reshape(mesh.shape[0], -1).float()
    return mesh

def mean_cov_map(input_set, mesh):
    input_set = input_set.transpose(-2,-1).unsqueeze(-1)
    delta = input_set - mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    wFM = torch.matmul(delta, delta.transpose(-1,-2))/mesh.shape[-1]
    wFM = wFM.view(input_set.shape[0], input_set.shape[1], input_set.shape[2], -1)
    wFM = torch.cat([input_set[...,0],wFM], dim = -1)
    return wFM.transpose(-1,-2)

def load_data(data_dir, batch_size, shuffle=True, num_workers=4):
    train_data = h5py.File(data_dir + ".hdf5" , 'r')
    xs = np.array(train_data['data'])
    ys = np.array(train_data['labels'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle, num_workers=num_workers)
    train_data.close()
    return train_loader_dataset


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims, num_channels)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    B, N, D, C = point_cloud.size()
    point_cloud = point_cloud.permute(0, 3, 1, 2) # [B, C, N, D]
    
    x_norm = (point_cloud ** 2).sum(-1).view(B, C, -1, 1) # [B, C, N, 1]
    y_norm = x_norm.view(B, C, 1, -1) # [B, C, 1, N]

    dist = x_norm + y_norm - 2.0 * torch.matmul(point_cloud, point_cloud.transpose(2, 3)) # [B, C, N, N]
    dist = dist.sum(1) # [B, N, N]
    
    return dist


def down_sampling(X, v, out_pts):
    B, N, _ = X.shape
    
    ind_all = []
    for b in range(B):
        indices = torch.multinomial(v[b], out_pts, replacement = False)
        ind_all.append(indices)
        
    ind_all = torch.stack(ind_all)
    idx = (torch.arange(B)*N).cuda()
    idx = idx.view((B, 1))
    k2 = ind_all + idx
    X = X.view(-1, X.shape[-1])
    return X[k2]


def knn(adj_matrix, k=20, include_myself=False):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    if include_myself:
        nn_idx = np.argsort(adj_matrix.cpu().detach().numpy(), axis=-1)[:,:,:k]
    else:
        nn_idx = np.argsort(adj_matrix, axis=-1)[:,:,1:k+1] #torch.nn.top_k(neg_adj, k=k)
    return nn_idx

def sample_subset(idx_input, num_output):
    return np.random.choice(idx_input, num_output ,replace = False)




















# END