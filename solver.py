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
            inputs = utils.sdt(inputs, grid, sigma)
            inputs = inputs * inputs
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
    model = ManifoldNet(10, params['num_neighbors'], params['num_points']).cuda()
    #model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    print("Start Training")

    # results = dict()
    # bins = np.linspace(0, 3.14 / 2, 10)

    # Iterate by Epoch
    for epoch in range(params['num_epochs']):  # loop over the dataset multiple times
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):

            print("--> Variable Setting up")

            # print("===\n{}\n===".format(inputs))

            # Variable Setup
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optim.zero_grad()

            print("--> Running Model")

            # Model Input/Output
            inputs = utils.sdt(inputs, params['grid'], params['sigma'])


            # norm = torch.norm(inputs, p=2, dim=3) # B, N, C
            # first_element = inputs[..., 0]
            # thetas = torch.acos(torch.clamp(norm * first_element, -1, 1))

            # print("[Thetas]: {}".format(thetas.size()))

            # for i in range(len(thetas)):
                # theta = thetas[i]
                # label = str(labels[i])

                # theta = theta.cpu().numpy()
                # theta = theta.flatten()

                # if label in results.keys():
                    # results[label].extend(theta)
                # else:
                    # results[label] = list(theta)


            # if len(results.keys()) >= 5 and batch_idx >= 5:

                # print("====")
                # for label in results:
                    # print("\nClass {}".format(label))
                    # theta = results[label]
                    # result, _ = np.histogram(theta, bins=bins)
                    # for i in range(len(result)):
                        # percentage = result[i] / sum(result)
                        # print("{}: {}%".format(round(bins[i], 3), round(percentage, 4)))

                # print("====\nNumber of Classes: {}".format(len(results)))
                # return

            #inputs = inputs
            # print(inputs.shape)
            outputs = model(inputs)

            print("--> Updating Model")

            # Update Loss and Do Backprop
            print(labels.squeeze().shape)
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
            #if batch_idx % params['log_interval'] == 0:
             #   acc = eval(test_iterator, model, params['grid'], params['sigma'])
              #  print("Accuracy: [{}]\n".format(acc))

        #acc = eval(test_iterator, model, grid, sigma)
        acc = 0.0
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
    
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path',     default='./mnistPC', type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',    default=15 ,          type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--num_epochs',    default=200 ,         type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--num_points',    default=512 ,         type=int,   metavar='N',   help='Number of points in a image')
    parser.add_argument('--log_interval',  default=10 ,          type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--grid',          default=5 ,           type=int,   metavar='N',   help='grid of sdt')
    parser.add_argument('--sigma',         default=0.5 ,         type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--log_dir',       default="./log_dir",  type=str,   metavar='N',   help='directory for logging')
    parser.add_argument('--baselr',        default=0.05 ,        type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--gpu',           default='1',        type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--num_neighbors', default=15,           type=int,   metavar='XXX', help='Number of Neighbors')

    args = parser.parse_args()

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