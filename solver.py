from model import ManifoldNet
import dataloader
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import utils
import h5py
from pdb import set_trace as st
import argparse
from logger import Logger

#trainloader = dataloader.getLoader("./mnistPC/train.hdf5", 80, 'train')

#optim = torch.optim.SGD(model.parameters(), lr=1e-6)

def eval(test_iterator, model, grid, sigma):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            inputs = utils.sdt(inputs, grid, sigma)
            inputs = inputs*inputs            
            adj = utils.pairwise_distance(inputs)
            outputs = model(inputs, adj) 
            outputs = torch.argmax(outputs, dim=-1)
            acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
        else:
            return np.mean(np.array(acc_all))



def train(train_data_dir, test_data_dir, num_epochs, log_interval, grid, sigma, batch_size, log_dir, baselr, gpu, neighbors):
    
    # Logger Setup and OS Configuration
    logger=Logger(log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    # Model Setup
    model = ManifoldNet(10, neighbors, 512).cuda()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    # Load Data
    test_iterator = utils.load_data(test_data_dir, batch_size=batch_size)
    train_iterator = utils.load_data(train_data_dir, batch_size=batch_size)
    
    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=baselr)
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Iterate by Epoch
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):
            
            # Variable Setup
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optim.zero_grad()
            
            # Neighbor Data Preperation
            adj = utils.pairwise_distance(inputs)
            knn_matrix = utils.knn(adj, k=neighbors, include_myself=True)
            knn_matrix = torch.Tensor(knn_matrix).long()
            
            # Model Input/Output
            inputs = utils.sdt(inputs, grid, sigma)
            inputs = inputs*inputs
            outputs = model(inputs, knn_matrix)
            
            # Update Loss and Do Backprop
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()
            running_loss.append(loss.item())
            
            # Update Loss Per Batch
            print("Batch: [{batch}/{total_batch}] Epoch: [{epoch}] Loss: [{loss}]".format(batch=batch_idx,
                                                                                         total_batch=len(train_iterator),
                                                                                         epoch=epoch,
                                                                                         loss=np.mean(running_loss)))
            
            # Periodically Show Accuracy
            if batch_idx % log_interval == 0:
                acc = eval(test_iterator, model, grid, sigma)
                print("Accuracy: [{}]\n".format(acc))
  
        acc = eval(test_iterator, model, grid, sigma)
        print("Epoch: [{epoch}/{total_epoch}] Loss: [{loss}] Accuracy: [{acc}]".format(epoch=epoch,
                                                                                      total_epoch=num_epochs,
                                                                                      loss=np.mean(running_loss),
                                                                                      acc=acc))
        logger.scalar_summary("running_loss", np.mean(running_loss), epoch)
        logger.scalar_summary("accuracy", acc, epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, '_'.join(["manifold", str(epoch + 1)])))

    print('Finished Training')
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path',    default='./mnistPC',  type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',   default=10 ,          type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--max_epoch',    default=200 ,         type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--log_interval', default=10 ,          type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--grid',         default=5 ,           type=int,   metavar='N',   help='grid of sdt')
    parser.add_argument('--sigma',        default=0.1 ,         type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--log_dir',      default="./log_dir",  type=str,   metavar='N',   help='directory for logging')
    parser.add_argument('--baselr',       default=0.01 ,        type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--gpu',          default='3,2',        type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--neighbors',    default=15,           type=int,   metavar='XXX', help='Number of Neighbors')

    args = parser.parse_args()
    test_data_dir = os.path.join(args.data_path, "test.hdf5")
    train_data_dir = os.path.join(args.data_path, "train.hdf5")
    train(train_data_dir, test_data_dir, args.max_epoch, args.log_interval, args.grid, args.sigma, args.batch_size, args.log_dir, args.baselr, args.gpu, args.neighbors)
