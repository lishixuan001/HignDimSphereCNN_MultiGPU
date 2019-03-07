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
        self.wFM1 = wFM.wFMLayer(1, 20, num_neighbor, num_points).cuda()
        self.wFM2 = wFM.wFMLayer(20, 40, num_neighbor, int(num_points)).cuda()
        self.wFM3 = wFM.wFMLayer(40, 60, num_neighbor, int(num_points)).cuda()
        self.last = wFM.Last(60, num_classes).cuda()
        
        ##DENSITY##
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm2d(60)
        ###########

    def forward(self, x, neighborhood_matrix):
        ##TANGENT PLANE##
#     	fm1 = self.wFM1(x, neighborhood_matrix)
#         fm2 = self.wFM2(fm1)
#         fm3 = self.wFM3(fm2)
        #################
        
        ##DENSITY##
        fm1 = self.wFM1(x, neighborhood_matrix)
        fm1 = F.relu(fm1)
        fm1 = fm1.permute(0, 3, 2, 1)
        fm1 = self.bn1(fm1)
        fm1 = fm1.permute(0, 3, 2, 1)
        #print(fm1.shape)
    	fm2 = self.wFM2(fm1)
        fm2 = F.relu(fm2)
        fm2 = fm2.permute(0, 3, 2, 1)
        fm2 = self.bn2(fm2)
        fm2 = fm2.permute(0, 3, 2, 1)
    	fm3 = self.wFM3(fm2)
        fm3 = F.relu(fm3)
        fm3 = fm3.permute(0, 3, 2, 1)
        fm3 = self.bn3(fm3)
        fm3 = fm3.permute(0, 3, 2, 1)
        ###########
        fm3 = fm3.contiguous()
        out = self.last(fm3)
        return out

