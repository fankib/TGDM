
import os

print(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import matplotlib.pyplot as plt
import argparse

from experiment import local_machine, Logger
from dataset import Dataset
from network import ResnetClassifier

from tgdm import TGDM
from tgdm.regularization import RegularizationFactory

''' fix seed '''
if local_machine():
    torch.manual_seed(1587039270)    

''' command line '''
parser = argparse.ArgumentParser()
parser.add_argument('--group', default='', type=str, help='Wandb group')
parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
parser.add_argument('--batch_size', default=256, type=int, help='the batch size to use')
parser.add_argument('--iterations', default=10000, type=int, help='amount of iterations to train over all')
parser.add_argument('--inner_iters', default=2, type=int, help='amount of inner iterations')
parser.add_argument('--train_split', default=0.9, type=float, help='The percentage of the available trainig data to use')
parser.add_argument('--valid_split', default=1.0, type=float, help='The percentage of the remaining trainig data to use for validation')
parser.add_argument('--architecture', default='resnet18', type=str, help='The network architecture to use')
parser.add_argument('--optimizer', default='tgdm', help='optimizer to use: [hd-tgdm, hdc-tgdm, t1t2-tgdm, torch-sgd, torch-adam]')
parser.add_argument('--dataset', default=Dataset.CIFAR10, type=str, help='The dataset to use: [cifar10, cifar100, birds]')
parser.add_argument('--data_augmentation', default='False', type=str, help='activate data augmentation')
parser.add_argument('--regulation', default='L2', type=str, help='the used regulation method')

# tune hyper training:
parser.add_argument('--hlr_lr', default=0.01, type=float, help='hyper learning rate for lr')
parser.add_argument('--hlr_momentum', default=0.01, type=float, help='hyper learning rate for momentum')
parser.add_argument('--hlr_regularization', default=0.01, type=float, help='hyper learning rate for regularization')

# initial values:
parser.add_argument('--lr', default=0.01, type=float, help='the initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, help='the initial momentum')
parser.add_argument('--regularization', default=0.01, type=float, help='the initial regularization weight')

# parse!
args = parser.parse_args()

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
model = ResnetClassifier(args.architecture, dataset.n_classes(), pretrained).to(device) # TODO: fix double device

''' train loop '''
regulation = RegularizationFactory().by_name(args.regulation)
defaults = {'hyper_learning_rate': [args.hlr_lr, args.hlr_momentum, args.hlr_regularization], 'lr': args.lr, 'momentum': args.momentum, 'regularization': args.regularization}
optimizer = TGDM(model.parameters(), defaults, regulation, logger)
valid_iterator = None
train_iterator = None
train_available = 0
for i in range(args.iterations):
    
    # reset validation iter
    if i % (len(valid_loader)) == 0:
        valid_iterator = iter(valid_loader)
    
    model.train()
    
    # manage available train batches
    if train_available - args.inner_iters < 0:
        print('refresh training iterator')
        train_iterator = iter(train_loader)
        train_available = len(train_loader)
    train_available -= args.inner_iters
    
    # TGDM Loop:
    optimizer.hyper_zero_grad()
    # inner iterations
    for t in range(args.inner_iters):
        logger.step()            
        optimizer.zero_grad()
        C = model.loss(train_iterator, device)
        C.backward()
        optimizer.step()        
        logger.log({'train cost': C.item()})
    # HO step:
    optimizer.zero_grad()
    E = model.loss(valid_iterator, device)
    E.backward()
    optimizer.hyper_step()
    logger.log({'valid energy': E.item()})

    # evaluate every 100 iters:
    if (i+1) % 100 == 0:
        total, correct, avg_loss = evaluate(model, train_loader, device)
        logger.log_evaluation('train', i, total, correct, avg_loss)
        total, correct, avg_loss = evaluate(model, valid_loader, device)
        logger.log_evaluation('valid', i, total, correct, avg_loss)
        total, correct, avg_loss = evaluate(model, test_loader, device)
        logger.log_evaluation('test', i, total, correct, avg_loss)

# done!