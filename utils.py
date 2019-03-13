import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import argparse

def load_args():
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path',     default='../mnistPC', type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',    default=15 ,          type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--num_epochs',    default=200 ,         type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--num_points',    default=512 ,         type=int,   metavar='N',   help='Number of points in a image')
    parser.add_argument('--log_interval',  default=10 ,          type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--grid',          default=15 ,          type=int,   metavar='N',   help='grid of sdt')
    parser.add_argument('--sigma',         default= 0.01,       type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--log_dir',       default="./log_dir",  type=str,   metavar='N',   help='directory for logging')
    parser.add_argument('--baselr',        default=0.05 ,        type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--gpu',           default='1',          type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--num_neighbors', default=15,           type=int,   metavar='XXX', help='Number of Neighbors')

    args = parser.parse_args()
    return args

def generate_spherical_grid(num_directions):
    


def data_generation(inputs, grid_size, sigma):
    batch_size, num_points, dimension_size = inputs.size() # [B * N * D]

    print("--> Normalizing Raw Data")
    inputs = raw_data_normalization(inputs)

    print("--> Mapping and Normalization")
    grid = grid_generation(grid_size)
    inputs = map_and_norm(inputs, grid, sigma)

    return inputs


def raw_data_normalization(inputs):
    """
    inputs: B * N * D
    """
    inputs = nn.functional.normalize(inputs, p=1, dim=2, eps=1e-10)
    return inputs
    

def grid_generation(grid_size):
    linspace = np.linspace(-1, 1, grid_size)
    grid = np.meshgrid(linspace, linspace)  # (2, grid_size, grid_size)
    grid = torch.from_numpy(np.array(grid))
    grid = grid.reshape(grid.size()[0], -1).float()  # (2, grid_size^2)
    return grid.cuda()

def map_and_norm(tensor_dataset, grid, sigma):
    tensor_dataset_spread = tensor_dataset.unsqueeze(-1)  # (data_size, num_points, 2, 1)
    tensor_dataset_spread = tensor_dataset_spread.repeat(
        (1, 1, 1, grid.size()[-1]))  # (data_size, num_points, 2, grid_size^2)
    grid_spread = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, grid_size^3)
    tensor_dataset_spread = tensor_dataset_spread - grid_spread  # (data_size, num_points, 2, grid_size^2)
    tensor_dataset_spread_transpose = tensor_dataset_spread.transpose(2, 3)  # (data_size, num_points, grid_size^2, 2)
    tensor_dataset_spread_transpose_norms = torch.norm(tensor_dataset_spread_transpose, dim=3, p=2,
                                                       keepdim=True)  # (data_size, num_points, grid_size^2, 1)
    tensor_dataset = torch.div(tensor_dataset_spread_transpose_norms,
                               -2.0 * np.power(sigma, 2))  # (data_size, num_points, grid_size^2, 1)
    tensor_dataset = torch.exp(tensor_dataset)  # (data_size, num_points, grid_size^2, 1)
    tensor_dataset = tensor_dataset.squeeze(-1)  # (data_size, num_points, grid_size^2)

    """ Normalization (Mapping) """
    tensor_dataset = nn.functional.normalize(tensor_dataset, p=2, dim=2, eps=1e-10)

    return tensor_dataset.unsqueeze(-1)


def sdt(inputs, grid, sigma):
    x = inputs
    dim = x.shape[2]
    num_point = x.shape[1]

    x_mean = torch.mean(x, dim=1, keepdim=True)
    x = (x - x_mean)
    x = x/torch.norm(x, dim=1, keepdim=True)
    
    #x_std = torch.std(x, dim=1, keepdim=True)
    #x = (x - x_mean)/(x_std+0.0001)
    #x_min = torch.min(x, dim=1, keepdim=True)[0]
    #x_max = torch.max(x, dim=1, keepdim=True)[0]
    #x = (x - x_min)/(x_max - x_min)
    
    linspace = np.linspace(-1.,1.,grid)
    mesh = linspace

    for i in range(dim-1):
        mesh = np.meshgrid(mesh, linspace)
    mesh = torch.from_numpy(np.array(mesh))#.cuda()
    mesh = mesh.reshape(mesh.shape[0], -1).float()

    temp = x.unsqueeze(-1).repeat( 1,1,1,mesh.shape[-1])
    temp = temp - mesh.unsqueeze(0).unsqueeze(0).cuda()#torch.from_numpy(np.expand_dims(np.expand_dims(mesh, 0),0)).cuda()

    out = torch.sum(temp**2, -2)
    out = torch.exp(-out / (2 * sigma**2 + 1e-10))
    norms = torch.sum(out, dim = 2, keepdim=True)
    out = (out/norms).unsqueeze(-1)
    return out

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
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0] #point_cloud.get_shape().as_list()[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0) #torch.expand_dims(point_cloud, 0)

    if len(point_cloud.shape) == 4:
        a, b, c, d = point_cloud.shape
        point_cloud = point_cloud.view(a, b, c*d)
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    #torch.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = torch.sum( point_cloud**2, dim=-1, keepdim = True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1) #torch.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def down_sampling(X, v, out_pts):
    B = X.shape[0]
    N = X.shape[1]
    #lst = torch.Tensor(list(range(N)))
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

def GumblerSinkhorn(input_set, u, v, times=20):
    e_weight = u * v.transpose(1, 0)
    e_weight = torch.exp(e_weight)
    for i in range(times):
        e_weight = e_weight / torch.sum(e_weight, dim=1, keepdim = True)
        e_weight = e_weight / torch.sum(e_weight, dim=0, keepdim = True)
    return torch.matmul(e_weight, input_set)

def knn(adj_matrix, k=20, include_myself = False):
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