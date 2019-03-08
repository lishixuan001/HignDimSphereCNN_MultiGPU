import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbor, num_points):
        super(ManifoldNet, self).__init__()
        self.wFM1 = wFM.wFMLayer(1, 30, num_neighbor, num_points)
        self.wFM2 = wFM.wFMLayer(30, 30, num_neighbor, num_points)
        self.wFM3 = wFM.wFMLayer(30, 30, num_neighbor, num_points)
        self.wFM4 = wFM.wFMLayer(30, 30, num_neighbor, num_points)
        self.wFM5 = wFM.wFMLayer(30, 30, num_neighbor, num_points)  
        
        
#         self.wFM2 = wFM.wFMLayer(30, 30, num_neighbor, int(num_points), 0.5)
#         self.wFM3 = wFM.wFMLayer(30, 30, num_neighbor, int(num_points*0.5), 0.5)
#         self.wFM4 = wFM.wFMLayer(30, 30, num_neighbor, int(int(num_points*0.5)*0.5), 0.5)
#         self.wFM5 = wFM.wFMLayer(30, 30, num_neighbor, int(int(int(num_points*0.5)*0.5)*0.5), 0.5)               
        self.last = wFM.Last(30, num_classes)
        
        ##DENSITY##
        self.nl = wFM.Nonlinear()
        ###########

    def forward(self, x, neighborhood_matrix):
        ##TANGENT PLANE##
#     	fm1 = self.wFM1(x, neighborhood_matrix)
#         fm2 = self.wFM2(fm1)
#         fm3 = self.wFM3(fm2)
        #################
        
        ##DENSITY##
        fm1 = self.wFM1(x, neighborhood_matrix)
        fm1 = self.nl(fm1)
        fm2 = self.wFM2(fm1, neighborhood_matrix)
        fm2 = self.nl(fm2)
        fm3 = self.wFM3(fm2, neighborhood_matrix)
        fm3 = self.nl(fm3)
        fm4 = self.wFM4(fm3, neighborhood_matrix)
        fm4 = self.nl(fm4)
        fm5 = self.wFM5(fm4, neighborhood_matrix)
        fm5 = self.nl(fm5)
        ###########
        out = self.last(fm5)
        return out

