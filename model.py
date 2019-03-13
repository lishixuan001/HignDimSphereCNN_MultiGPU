import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import wFM
import utils
from pdb import set_trace as st

class ManifoldNet(nn.Module):
    def __init__(self, num_classes, num_neighbors, num_points):
        super(ManifoldNet, self).__init__()

        self.k = num_neighbors
        self.points = num_points
        self.wFM1_1 = wFM.wFMLayer1(5, num_neighbors, num_points)
        #print(num_points)
        self.wFM1_2 = wFM.wFMLayer2(5, 30)
        self.first_linear = nn.Linear(2, 10)
        self.second_linear = nn.Linear(10, 30)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.wFM2_1 = wFM.wFMLayer1(30, num_neighbors, num_points)
        self.wFM2_2 = wFM.wFMLayer2(30, 50)
        
        self.w1 = wFM.wFMLayer(1, 30, num_neighbors, num_points)
        self.w2 = wFM.wFMLayer(30, 40, num_neighbors, num_points)

        # self.wFM3_1 = wFM.wFMLayer1(30, num_neighbors, num_points)
        # self.wFM3_2 = wFM.wFMLayer2(30, 30)

        # self.wFM2 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM3 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM4 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        # self.wFM5 = wFM.wFMLayer(30, 30, num_neighbors, num_points)
        self.Last = wFM.Last(40, num_classes, 512)
        self.ltest = nn.Linear(1024, 1280*60)
        #self.NonLinear = wFM.Nonlinear()
        self.sig = nn.Parameter(torch.ones(num_points)*0.01)

    def forward(self, inputs):
            
        
        inputs = utils.sdt(inputs, 5, self.sig)
        print(1 in torch.isnan(inputs).detach().cpu().numpy())
        print(self.sig)
        if 1 in torch.isnan(inputs).detach().cpu().numpy():
            st()
#         print(self.sig)
#         print("######")
#         print(inputs)
#         st()
        ##st()
#         input_shape = inputs.shape
#         print(input_shape)
#         inputs = inputs.view(-1, 1024)
#         inputs = self.ltest(inputs).view(-1, 512, 30, 5)
#         print(input_shape)
# #         inputs = self.softmax(self.second_linear(self.relu(self.first_linear(inputs.view(-1,2)))).view(input_shape[0],input_shape[1],-1)).unsqueeze(-1)
        
#         print(inputs.shape)
#         #inputs = inputs/torch.sum(inputs, dim=2, keepdim=True)
        adj = utils.pairwise_distance(inputs)
        knn_matrix = utils.knn(adj, k=self.k, include_myself=True)
        knn_matrix = torch.Tensor(knn_matrix).long()

        fm1 = self.w1(inputs, knn_matrix)
        #print(fm1)
#         #fm1 = self.NonLinear(fm1)


        adj = utils.pairwise_distance(fm1)
        knn_matrix = utils.knn(adj, k=self.k, include_myself=True)
        knn_matrix = torch.Tensor(knn_matrix).long()
        
        
#         fm2 = self.wFM2_1(fm1, knn_matrix)
#         fm2 = self.wFM2_2(fm2)
#         #fm2 = self.NonLinear(fm2)
#         qn = torch.norm(fm2, p=2, dim=2, keepdim = True)
#         fm2 = fm2.div(qn)
#         print(fm2.shape)
        # fm3 = self.wFM3(fm2, knn_matrix)
        # fm3 = self.NonLinear(fm3)

        # fm4 = self.wFM4(fm3, knn_matrix)
        # fm4 = self.NonLinear(fm4)

        # fm5 = self.wFM5(fm4, knn_matrix)
        # fm5 = self.NonLinear(fm5)
        fm2 = self.w2(fm1, knn_matrix)
        out = self.Last(fm2)

#         print("===========================")
#         print("[Output]")
#         print("Size: {}".format(out.size()))
#         print("Tensor: {}".format(out))
#         print("===========================")
        return out

