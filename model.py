import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbor):
        super(ManifoldNet, self).__init__()
        self.wFM1 = wFM.wFMLayer(1, 32, num_neighbor).cuda()
        self.wFM2 = wFM.wFMLayer(32, 128, num_neighbor).cuda()
        self.wFM3 = wFM.wFMLayer(128, 256, num_neighbor).cuda()
        self.last = wFM.Last(256, num_classes).cuda()
    
    def forward(self, x, neighborhood_matrix):
        return self.last(self.wFM3(self.wFM2(self.wFM1(x, neighborhood_matrix), neighborhood_matrix), neighborhood_matrix))
        #return self.last(self.wFM1(x, adj))

