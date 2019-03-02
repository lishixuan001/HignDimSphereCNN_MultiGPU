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
        self.wFM1 = wFM.wFMLayer(1, 10, num_neighbor, num_points).cuda()
        self.wFM2 = wFM.wFMLayer(10, 20, num_neighbor, int(num_points), 0.8).cuda()
        self.wFM3 = wFM.wFMLayer(20, 40, num_neighbor, int(num_points*0.8), 0.7).cuda()
        self.last = wFM.Last(40, num_classes).cuda()

    def forward(self, x, neighborhood_matrix):
    	fm1 = self.wFM1(x, neighborhood_matrix)
    	fm2 = self.wFM2(fm1)
    	fm3 = self.wFM3(fm2)
    	#print(self.wFM1.w1[0])
        return self.last(fm3)
        #return self.last(self.wFM1(x, adj))

