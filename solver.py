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

def eval(test_iterator, model, mesh):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            outputs = model(inputs.unsqueeze(-1), mesh)
            outputs = torch.argmax(outputs, dim=-1)
            acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
        else:
            return np.mean(np.array(acc_all))



def train(train_data_dir, test_data_dir, train_iter, log_interval, grid, sigma, batch_size, log_dir, baselr, gpu):
    logger=Logger(log_dir)
    mesh = utils.mesh_mat(grid)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    model = ManifoldNet(10, 15, 512).cuda()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=baselr)
    test_iterator = utils.load_data(test_data_dir, batch_size=25)
    train_iterator = utils.load_data(train_data_dir, batch_size=batch_size)
    t = len(train_iterator)
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    for epoch in range(train_iter):  # loop over the dataset multiple times
        running_loss = []
        start = time.time()
        for i, (inputs, labels) in enumerate(train_iterator):
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optim.zero_grad()
            adj = utils.pairwise_distance(inputs)
            inputs = utils.sdt(inputs, grid, sigma)
            inputs = inputs*inputs
            outputs = model(inputs, adj)
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()
            running_loss.append( loss.item() )
            print("Batch: "+str(i)+"/"+str(t)+" Epoch: "+str(epoch)+" Loss: "+str(np.mean(running_loss) ))
            if i % 60 == 10:
                acc = eval(test_iterator, model, mesh)
                print("Accuracy: "+str(acc))
        end = time.time()
        #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        acc = eval(test_iterator, model, mesh)
        print("Epoch: "+str(epoch)+" finished, took "+str(end-start)+" seconds with loss: "+str(np.mean(running_loss))+" acc: "+str(acc))
        logger.scalar_summary("running_loss", np.mean(running_loss), epoch)
        logger.scalar_summary("accuracy", acc, epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, '_'.join(["manifold", str(epoch + 1)])))

    print('Finished Training')
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path', default='./mnistPC',  type=str, metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size', default=8 , type=int, metavar='N', help='Batch size of test set')
    parser.add_argument('--max_epoch', default=200 , type=int, metavar='N', help='Epoch to run')
    parser.add_argument('--log_interval', default=10 , type=int, metavar='N', help='log_interval')
    parser.add_argument('--grid', default=10 , type=int, metavar='N', help='grid of sdt')
    parser.add_argument('--sigma', default=0.1 , type=float, metavar='N', help='sigma of sdt')
    parser.add_argument('--log_dir', default="./log_dir" , type=str, metavar='N', help='directory for logging')
    parser.add_argument('--baselr', default=0.01 , type=float, metavar='N', help='sigma of sdt')
    parser.add_argument('--gpu', default='5,2',  type=str, metavar='XXX', help='GPU number')

    args = parser.parse_args()
    test_data_dir = os.path.join(args.data_path, "test.hdf5")
    train_data_dir = os.path.join(args.data_path, "train.hdf5")
    train(train_data_dir, test_data_dir, args.max_epoch, args.log_interval, args.grid, args.sigma, args.batch_size, args.log_dir, args.baselr, args.gpu)
