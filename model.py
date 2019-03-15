import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbors, num_points, batch_size, grid_size):
        super(ManifoldNet, self).__init__()

        self.k = num_neighbors
        self.points = num_points
        
        self.wFw1 = wFM.wFMLayer(5, 30, num_neighbors, num_points)
        self.wFw2 = wFM.wFMLayer(30, 40, num_neighbors, num_points)
        
        self.NL1 = wFM.Nonlinear()
        self.NL2 = wFM.Nonlinear()

        # self.wFM2 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM3 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM4 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM5 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        
        self.Last = wFM.Last(40, num_classes, 512)
       
    def forward(self, inputs):
        
#         print("===\nInputs\n{}".format(inputs))
      
        adj = utils.pairwise_distance(inputs)
        knn_matrix = utils.knn(adj, k=self.k, include_myself=True)
        knn_matrix = torch.Tensor(knn_matrix).long()
        
        fm1 = self.wFw1(inputs, knn_matrix)
#         print("===\nfm1-output\n{}".format(fm1))
        
#         fm1 = self.NL1(fm1)
#         print("===\nfm1-nonlinear\n{}".format(fm1))
        
        fm2 = self.wFw2(fm1, knn_matrix)
#         print("===\nfm2-output\n{}".format(fm2))
        
#         fm2 = self.NL2(fm2)
#         print("===\nfm2-output\n{}".format(fm2))
        
        out = self.Last(fm2)
#         print("===\nout-output\n{}".format(out))
        
        # fm3 = self.wFM3(fm2, knn_matrix)
        # fm3 = self.NonLinear(fm3)

        # fm4 = self.wFM4(fm3, knn_matrix)
        # fm4 = self.NonLinear(fm4)

        # fm5 = self.wFM5(fm4, knn_matrix)
        # fm5 = self.NonLinear(fm5)


#         print("===========================")
#         print("[Output]")
#         print("Size: {}".format(out.size()))
#         print("Tensor: {}".format(out))
#         print("===========================")
        
        return out

