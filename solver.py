from model import ManifoldNet
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
import torch.utils.data

def eval(test_iterator, model, grid, sigma):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=-1)
            acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
        else:
            return np.mean(np.array(acc_all))


def train(params):

    print("Model Setting Up")

    # Logger Setup and OS Configuration
    logger = Logger(params['log_dir'])
    #os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']

    print("Loading Data")

    # Load Data
    test_iterator = utils.load_data(params['test_dir'], batch_size=params['batch_size'])
    train_iterator = utils.load_data(params['train_dir'], batch_size=params['batch_size'])

    # Model Setup
    model = ManifoldNet(10, params['num_neighbors'], params['num_points'], params['batch_size'], params['grid']).cuda()
    #model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    print("Start Training")

    # Iterate by Epoch
    for epoch in range(params['num_epochs']):  # loop over the dataset multiple times
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):

            """ Variable Setup """
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optim.zero_grad()

            """ Model Input/Output """
            inputs = utils.data_generation(inputs, params['grid'], params['sigma'])
            outputs = model(inputs)

            """ Update Loss and Do Backprop """ 
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
            if batch_idx % params['log_interval'] == 0:
               acc = eval(test_iterator, model, params['grid'], params['sigma'])
               print("Accuracy: [{}]\n".format(acc))

        acc = eval(test_iterator, model, grid, sigma)
        acc = eval(test_iterator, model, params['grid'], params['sigma'])
        print("Epoch: [{epoch}/{total_epoch}] Loss: [{loss}] Accuracy: [{acc}]".format(epoch=epoch,
                                                                                      total_epoch=params['num_epochs'],
                                                                                      loss=np.mean(running_loss),
                                                                                      acc=acc))
        logger.scalar_summary("running_loss", np.mean(running_loss), epoch)
        logger.scalar_summary("accuracy", acc, epoch)
        torch.save(model.state_dict(), os.path.join(params['log_dir'], '_'.join(["manifold", str(epoch + 1)])))

    print('Finished Training')
    logger.close()

if __name__ == '__main__':

    print("Loading Configurations")
    
    args = utils.load_args()

    params = dict(
        train_dir = os.path.join(args.data_path, "train"),
        test_dir  = os.path.join(args.data_path, "test"),
        num_points     = args.num_points,
        num_epochs     = args.num_epochs,
        log_interval   = args.log_interval,
        grid           = args.grid,
        sigma          = args.sigma,
        batch_size     = args.batch_size,
        log_dir        = args.log_dir,
        baselr         = args.baselr,
        gpu            = args.gpu,
        num_neighbors  = args.num_neighbors
    )
    

    train(params)