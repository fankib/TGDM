#!/usr/bin/env python

# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

### CHANGELOG: ###
#
# * Adapted to datasets and classifier from TGDM experiments.
#

### Note ###
# you require the adatune package before running:
# install with
# git clone https://github.com/awslabs/adatune.git
# pip install -e .

import argparse
import os

import torch.optim as optim

from adatune.data_loader import *
from adatune.mu_adam import MuAdam
from adatune.mu_sgd import MuSGD
from adatune.network import *
from adatune.utils import *

# own imports
import torch
import torch.nn as nn
from experiment import local_machine, Logger, Optimizer, StopWatch, ExperimentEvaluation
from dataset import Dataset
from network import ResnetClassifier

def cli_def():
    parser = argparse.ArgumentParser(description='CLI for running automated Learning Rate scheduler methods')
    # own args
    parser.add_argument('--group', default='', type=str, help='Wandb group')
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
    parser.add_argument('--batch_size', default=64, type=int, help='the batch size to use')
    parser.add_argument('--train_split', default=0.1, type=float, help='The percentage of the available trainig data to use')
    parser.add_argument('--valid_split', default=0.11, type=float, help='The percentage of the remaining trainig data to use for validation')
    parser.add_argument('--data_augmentation', default='False', type=str, help='activate data augmentation')
    parser.add_argument('--dataset', default=Dataset.CIFAR10, type=str, help='The dataset to use: [cifar10, cifar100, birds]')
    parser.add_argument('--validation_iteration', default=100, type=int, help='run test/valid/train accuracy after this amount of iterations')
    parser.add_argument('--iterations', default=10000, type=int, help='amount of iterations to train over all')
    parser.add_argument('--architecture', default='resnet18', type=str, help='The network architecture to use')
    
    # marthe args
    #parser.add_argument('--network', type=str, default='vgg', choices=['resnet', 'vgg'])
    #parser.add_argument('--dataset', type=str, choices=['cifar_10', 'cifar_100'], default='cifar_10')
    #parser.add_argument('--num-epoch', type=int, default=200)
    #parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--hyper-lr', type=float, default=1e-8)
    parser.add_argument('--alpha', type=float, default=1e-6)
    parser.add_argument('--model-loc', type=str, default='./marthe-model.pt')
    parser.add_argument('--grad-clipping', type=float, default=100.0)
    parser.add_argument('--mu', type=float, default=0.99999)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def train_rtho(args, optim_name, lr, momentum, wd, hyper_lr, alpha, model_loc,
               grad_clipping, first_order, seed, mu=1.0):
    
    # do not seed:
    # torch.manual_seed(seed)    

    ''' device '''
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu != -1 and torch.cuda.is_available() else 'cpu')
    use_data_augmentation = (args.data_augmentation == 'True')
    print('Use device:', device)
    
    ''' logger '''
    logger = Logger('tgdm', args.group, not local_machine())
    logger.log_args(args)
    
    ''' load dataset '''
    dataset = Dataset(args.dataset, args.batch_size, args.train_split, args.valid_split, use_data_augmentation)
    train_loader, valid_loader, test_loader = dataset.loaders()    
    
    ''' create classifier '''
    pretrained = (args.dataset == Dataset.BIRDS)
    if 'resnet' in args.architecture:
        model = ResnetClassifier(args.architecture, dataset.n_classes(), pretrained).to(device)

    evaluation = ExperimentEvaluation(logger)  
    watch = StopWatch()
    watch.start()

    # assign argparse parameters
    criterion = nn.CrossEntropyLoss().to(device)    
    cur_lr = lr
    timestep = 0

    #train_data, test_data = data_loader(network, dataset, batch_size)

    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
        hyper_optim = MuAdam(optimizer, hyper_lr, grad_clipping, first_order, mu, alpha, device)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        hyper_optim = MuSGD(optimizer, hyper_lr, grad_clipping, first_order, mu, alpha, device)

    vg = ValidationGradient(valid_loader, nn.CrossEntropyLoss(), device)
    for epoch in range(999):
        train_correct = 0
        train_loss = 0

        for inputs, labels in train_loader:
            model.train()
            timestep += 1
            logger.step()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            logger.log({'train cost': loss.item()})

            train_pred = outputs.argmax(1)
            train_correct += train_pred.eq(labels).sum().item()

            first_grad = ag.grad(loss, model.parameters(), create_graph=True, retain_graph=True)

            hyper_optim.compute_hg(model, first_grad)

            for params, gradients in zip(model.parameters(), first_grad):
                params.grad = gradients

            optimizer.step()
            E_grad, E = vg.val_grad(model)
            hyper_optim.hyper_step(E_grad)
            logger.log({'valid energy': E.item()})
            clear_grad(model)
            
            
            # log learning rate            
            logger.log({'lr': optimizer.param_groups[0]['lr'],\
                        'hyper_lr': hyper_optim.hyper_optim.param_groups[0]['lr']})
            
            if (timestep) % args.validation_iteration == 0:
                watch.pause()
                evaluation.evaluate(model, timestep, train_loader, valid_loader, test_loader, watch.current_seconds(), device)
                watch.resume()
            logger.log({'stopwatch': watch.current_seconds()})
            
            # exit after max iterations:
            if args.iterations == timestep:
                print('max iterations reached!')                
                return

        # use own evaluation!
        #train_acc = 100.0 * (train_correct / len(train_loader.dataset))
        #val_loss, val_acc = compute_loss_accuracy(model, test_data, criterion, device)
                
        #if val_acc > best_val_accuracy:
        #    best_val_accuracy = val_acc
        #    torch.save(net.state_dict(), model_loc)

        #print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
        #print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
        #print('best val_accuracy is : {}'.format(best_val_accuracy))

        


if __name__ == '__main__':
    args = cli_def().parse_args()
    print(args)

    if os.path.exists(args.model_loc):
        os.remove(args.model_loc)

    train_rtho(args, args.optimizer, args.lr, args.momentum,
               args.wd, args.hyper_lr, args.alpha, args.model_loc, args.grad_clipping, args.first_order, args.seed,
               args.mu)