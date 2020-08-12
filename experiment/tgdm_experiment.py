import sys
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import matplotlib.pyplot as plt
import argparse

from experiment import local_machine, Logger, Optimizer, StopWatch, ExperimentEvaluation
from dataset import Dataset
from network import ResnetClassifier, DensetNetClassifier, ConvNetClassifier

from tgdm import TGDM, TGDM_T1T2, TGDM_HD, TGDM_HDC, PYTORCH_SGD_STEP, PYTORCH_SGD_DEC
from tgdm.regularization import RegularizationFactory
from tgdm.tgdm_base import Buffer

''' fix seed '''
#if local_machine():
#    torch.manual_seed(1587039270)    

''' command line '''
parser = argparse.ArgumentParser()
parser.add_argument('--group', default='', type=str, help='Wandb group')
parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
parser.add_argument('--batch_size', default=256, type=int, help='the batch size to use')
parser.add_argument('--iterations', default=10000, type=int, help='amount of iterations to train over all')
parser.add_argument('--inner_iters', default=2, type=int, help='amount of inner iterations')
parser.add_argument('--outer_iters', default=1, type=int, help='amount of outer iterations')
parser.add_argument('--train_split', default=0.9, type=float, help='The percentage of the available trainig data to use')
parser.add_argument('--valid_split', default=1.0, type=float, help='The percentage of the remaining trainig data to use for validation')
parser.add_argument('--architecture', default='resnet18', type=str, help='The network architecture to use')
parser.add_argument('--optimizer', default='torch-sgd', help='optimizer to use: [tgdm-hd, tgdm-hdc, tgdm-t1t2, tgdm, torch-sgd, torch-adam]')
parser.add_argument('--dataset', default=Dataset.CIFAR10, type=str, help='The dataset to use: [cifar10, cifar100, birds]')
parser.add_argument('--data_augmentation', default='False', type=str, help='activate data augmentation')
parser.add_argument('--regulation', default='L2', type=str, help='the used regulation method')
parser.add_argument('--validation_iteration', default=100, type=int, help='run test/valid/train accuracy after this amount of iterations')

# tune hyper training:
parser.add_argument('--hlr_lr', default=0.01, type=float, help='hyper learning rate for lr')
parser.add_argument('--hlr_momentum', default=0.01, type=float, help='hyper learning rate for momentum')
parser.add_argument('--hlr_regularization', default=0.01, type=float, help='hyper learning rate for regularization')

# initial values:
parser.add_argument('--lr', default=0.01, type=float, help='the initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, help='the initial momentum')
parser.add_argument('--regularization', default=0.01, type=float, help='the initial regularization weight')

# classic lr decay (only torch-sgd)
parser.add_argument('--lr_decay_iterations', default=1000, type=int, help='divide the learning rate by 10 after this amount of iterations')
parser.add_argument('--lr_decay', default=0.99, type=float, help='multiply learning rate by this after each iteration')


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
#eval_train_loader, eval_valid_loader, eval_test_loader = dataset.loaders() # twice for evaluation

''' create classifier '''
pretrained = (args.dataset == Dataset.BIRDS)
if 'resnet' in args.architecture:
    model = ResnetClassifier(args.architecture, dataset.n_classes(), pretrained).to(device)
if 'densenet' in args.architecture:
    model = DensetNetClassifier(args.architecture, dataset.n_classes(), pretrained).to(device)
if 'convnet' in args.architecture:
    model = ConvNetClassifier(args.architecture, dataset.n_classes()).to(device)

''' training loops '''
watch = StopWatch()
evaluation = ExperimentEvaluation(logger)
watch.start()
regulation = RegularizationFactory().by_name(args.regulation)
defaults = {'hyper_learning_rate': [args.hlr_lr, args.hlr_momentum, args.hlr_regularization],\
            'lr': args.lr,\
            'momentum': args.momentum,\
            'regularization': args.regularization}

if args.optimizer in [Optimizer.TORCH_SGD, Optimizer.TORCH_SGD_DEC]:
    ''' classic torch SGD with decaying learning rate '''
    
    if args.optimizer == Optimizer.TORCH_SGD:
        optimizer = PYTORCH_SGD_STEP(model.parameters(), defaults, regulation, logger, args.lr_decay_iterations)
    elif args.optimizer == Optimizer.TORCH_SGD_DEC:
        optimizer = PYTORCH_SGD_DEC(model.parameters(), defaults, regulation, logger, args.lr_decay)
    
    for i in range(args.iterations):
        
        if i % (len(train_loader)) == 0:
            print('refresh training iterator')
            train_iterator = iter(train_loader)        
        
        logger.step()
        model.train()
        optimizer.zero_grad()
        C = model.loss(train_iterator, device)
        C.backward()
        optimizer.step()
        logger.log({'train cost': C.item()})
        
        # evaluate every 100 iters:
        if (i+1) % args.validation_iteration == 0:
            watch.pause()
            evaluation.evaluate(model, i+1, train_loader, valid_loader, test_loader, watch.current_seconds(), device)
            watch.resume()
        logger.log({'stopwatch': watch.current_seconds()})
        
    

if args.optimizer in [Optimizer.TGDM_HD, Optimizer.TGDM_HDC]:
    ''' TGDM HD/HDC Loop '''
    if args.optimizer == Optimizer.TGDM_HD:
        optimizer = TGDM_HD(model.parameters(), defaults, regulation, logger)    
    if args.optimizer == Optimizer.TGDM_HDC:
        optimizer = TGDM_HDC(model.parameters(), defaults, regulation, logger)    
    train_iterator = None    
    for i in range(args.iterations):        
        
        # manage available train batches
        if i % (len(train_loader)) == 0:
            print('refresh training iterator')
            train_iterator = iter(train_loader)        
        
        # TGDM HD Loop:        
        logger.step()     
        model.train()
        optimizer.zero_grad()
        C = model.loss(train_iterator, device)
        C.backward()
        optimizer.step()        
        logger.log({'train cost': C.item()})
    
        # evaluate every 100 iters:
        if (i+1) % args.validation_iteration == 0:
            watch.pause()
            evaluation.evaluate(model, i+1, train_loader, valid_loader, test_loader, watch.current_seconds(), device)
            watch.resume()
        logger.log({'stopwatch': watch.current_seconds()})

if args.optimizer == Optimizer.TGDM_T1T2:
    ''' TGDM T1T2 Loop '''
    optimizer = TGDM_T1T2(model.parameters(), defaults, regulation, logger)
    valid_iterator = None
    train_iterator = None    
    for i in range(args.iterations):        
        
        # reset validation iter
        if i % (len(valid_loader)) == 0:
            print('refresh validation iterator')
            valid_iterator = iter(valid_loader)
           
        # manage available train batches
        if i % (len(train_loader)) == 0:
            print('refresh training iterator')
            train_iterator = iter(train_loader)        
        
        # TGDM_T1T2 Loop:        
        optimizer.hyper_zero_grad()
        # inner iteration
        logger.step()            
        model.train()
        optimizer.zero_grad()
        C = model.loss(train_iterator, device)
        C.backward()
        optimizer.step()        
        logger.log({'train cost': C.item()})
        # HO step:
        model.eval()
        optimizer.zero_grad()
        E = model.loss(valid_iterator, device)
        E.backward()
        optimizer.hyper_step()
        logger.log({'valid energy': E.item()})
    
        # evaluate every 100 iters:
        if (i+1) % args.validation_iteration == 0:
            watch.pause()
            evaluation.evaluate(model, i+1, train_loader, valid_loader, test_loader, watch.current_seconds(), device)
            watch.resume()
        logger.log({'stopwatch': watch.current_seconds()})

if args.optimizer == Optimizer.TGDM:
    ''' TGDM Loop '''
    optimizer = TGDM(model.parameters(), defaults, regulation, logger)
    valid_iterator = None
    train_iterator = None
    valid_available = 0
    train_available = 0
    for i in range(int(args.iterations/args.inner_iters)):
        
        # TGDM Loop:        
        optimizer.hyper_zero_grad()        
        # inner iterations
        for t in range(args.inner_iters):
            model.train()
            
            # manage available train batches
            if train_available == 0:
                print('refresh training iterator')
                train_iterator = iter(train_loader)
                train_available = len(train_loader)
            train_available -= 1
            
            logger.step()            
            optimizer.zero_grad()
            C = model.loss(train_iterator, device)
            C.backward()
            optimizer.step()        
            logger.log({'train cost': C.item()})
            
            # debug:
            if np.isnan(C.item()):
                sys.exit(1)
            
            # evaluate every 100 iters:
            if (i*args.inner_iters+t+1) % int(args.validation_iteration) == 0:
                watch.pause()
                evaluation.evaluate(model, i+1, train_loader, valid_loader, test_loader, watch.current_seconds(), device)
                watch.resume()
            logger.log({'stopwatch': watch.current_seconds()})
        # HO step:
        model.eval()
        Es = []
        for t in range(args.outer_iters):
                      
            # manage available valid batches
            if valid_available == 0:
                print('refresh validation iterator')
                valid_iterator = iter(valid_loader)
                valid_available = len(valid_loader)
            valid_available -= 1
            
            #logger.step()
            optimizer.zero_grad()
            E = model.loss(valid_iterator, device)
            E.backward()
            optimizer.hyper_step()
            Es.append(E.item())
        logger.log({'valid energy': np.mean(Es)})
        
        # debug
        print({'lr': optimizer.param_groups[0][Buffer.learning_rate].get_value(),\
                    'momentum': optimizer.param_groups[0][Buffer.momentum].get_value(),\
                    'regularization': optimizer.param_groups[0][Buffer.regularization].get_value(),\
                    })
# done!