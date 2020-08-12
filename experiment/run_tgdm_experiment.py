import inspect
import copy

from tgdm.parametrized_value import RandomUniformValue, ExpValue, SigmoidValue, ParametrizedValue

from experiment import local_machine, ExperimentFactory, Optimizer, TGDMExperimentArgBuilder
from dataset import Dataset

# say my name:
name = lambda : inspect.stack()[1][3]

def main():
    concurrent_tasks = 1 if local_machine() else 2
    factory = ExperimentFactory(concurrent_tasks, 'tgdm_experiment.py')
        
    # experiments
    #cifar10_tgdm(factory)    
    
    # gridsearch learning rates
    #cifar10_sgd_gs(factory)
    #cifar10_tgdm_hdc_gs_learningrates(factory) # heavy overfitting    
    
    # gridsearch learning rates:
    #cifar10_tgdm_gs_lr(factory)
    
    # estimate params:
    #find_lr_reg_momentum(factory)
    
    # robustness scenarios
    #scenario_a_reuse(factory)
    #scenario_b_random_search(factory)
    
    # after beta bugfix
    #cifar10_tgdm_bufix(factory)
    #birds_tgdm(factory)
    
    # td vs t1t2 vs tgdm
    cifar10_tgdm_hd_t1t2_rerun(factory)
    
    # robustness scenarion + random search
    #scenario_b_random_search_cifar10_10(factory)

def scenario_b_random_search_cifar10_10(factory):
    ''' runs a large random search and tries to converge '''
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(5000).group(name())
    
    # unknown settings
    defaults.architecture('resnet18')
    defaults.dataset(Dataset.CIFAR10)
    defaults.data_augmentation(True)
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)
    defaults.validation_iteration(300)
    
    # tgdm
    defaults_tgdm = copy.deepcopy(defaults)
    defaults_tgdm.inner_iters(39).optimizer(Optimizer.TGDM).regulation('L2')
    defaults_tgdm.hlr_lr(0.001).hlr_momentum(0.001).hlr_regularization(0.0001)
    
    # sgd_step
    defaults_sgd = copy.deepcopy(defaults)    
    defaults_sgd.optimizer(Optimizer.TORCH_SGD).regulation('L2')
    
    # sgd_dec
    defaults_sgd_dec = copy.deepcopy(defaults)
    defaults_sgd_dec.optimizer(Optimizer.TORCH_SGD_DEC).regulation('L2')
    
    # run random search:
    for i in range(100):
        lr = RandomUniformValue(0.01, 1.0, ExpValue()).get_value().item()
        momentum = RandomUniformValue(0.1, 0.9, SigmoidValue()).get_value().item()
        regularization = RandomUniformValue(0.00001, 1.0, ExpValue()).get_value().item()
        print('use random entry: ', lr, momentum, regularization)
        lr_step = int(RandomUniformValue(1000, 4000, ParametrizedValue()).get_value().item())
        lr_decay = RandomUniformValue(0.99, 1.0, ParametrizedValue()).get_value().item()
        print('use sgd step', lr_step, 'and lr decay:', lr_decay)
        
        # run sgd_step
        args_sgd = copy.deepcopy(defaults_sgd)
        args_sgd.gpu(gpu.next())
        args_sgd.lr_decay_iterations(lr_step)
        args_sgd.lr(lr).momentum(momentum).regularization(regularization)
        r.run(args_sgd.build())
        
        # run sgd_dec
        args_sgd_dec = copy.deepcopy(defaults_sgd_dec)
        args_sgd_dec.gpu(gpu.next())
        args_sgd_dec.lr_decay(lr_decay)
        args_sgd_dec.lr(lr).momentum(momentum).regularization(regularization)
        r.run(args_sgd_dec.build())
        
        # run tgdm
        args_tgdm = copy.deepcopy(defaults_tgdm)
        args_tgdm.gpu(gpu.next())
        args_tgdm.lr(lr).momentum(momentum).regularization(regularization)
        r.run(args_tgdm.build())
    
    # wait for it
    r.wait()

def cifar10_tgdm_hd_t1t2_rerun(factory):
    ''' runs a large random search and tries to converge '''
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(7000).group(name())
    
    # dataset settings
    defaults.architecture('resnet18')
    defaults.dataset(Dataset.CIFAR10)
    defaults.data_augmentation(True)
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)    
    defaults.validation_iteration(100)
    
    # initial hyperparams
    defaults.lr(0.03).momentum(0.5).regularization(0.1)        
    
    # tgdm
    defaults_tgdm = copy.deepcopy(defaults)
    defaults_tgdm.optimizer(Optimizer.TGDM).regulation('L2')
    defaults_tgdm.hlr_lr(0.001).hlr_momentum(0.001).hlr_regularization(0.0001)
    
    # hd
    defaults_hd = copy.deepcopy(defaults)    
    defaults_hd.optimizer(Optimizer.TGDM_HD).regulation('L2')
    defaults_hd.hlr_lr(0.0001).hlr_momentum(0.0001).hlr_regularization(0.0001)    
    
    # hdc
    defaults_hdc = copy.deepcopy(defaults)    
    defaults_hdc.optimizer(Optimizer.TGDM_HDC).regulation('L2')
    defaults_hdc.hlr_lr(0.001).hlr_momentum(0.001).hlr_regularization(0.0001)    
    
    # t1t2
    defaults_t1t2 = copy.deepcopy(defaults)    
    defaults_t1t2.optimizer(Optimizer.TGDM_T1T2).regulation('L2')
    defaults_t1t2.hlr_lr(0.0001).hlr_momentum(0.0001).hlr_regularization(0.0001)    
    
    # sgd
    defaults_sgd = copy.deepcopy(defaults)
    defaults_sgd.optimizer(Optimizer.TORCH_SGD).regulation('L2')
    defaults_sgd.lr_decay_iterations(2070)
    
    # sgd_dec
    defaults_sgd_dec = copy.deepcopy(defaults)
    defaults_sgd_dec.optimizer(Optimizer.TORCH_SGD_DEC).regulation('L2')
    defaults_sgd_dec.lr_decay(0.997)    
    
    for i in range(10):   
        
        
        # run sgd*
        args_sgd = copy.deepcopy(defaults_sgd)
        args_sgd.gpu(gpu.next())
        args_sgd.lr(0.0324).momentum(0.25).regularization(0.06566).lr_decay_iterations(2070)
        r.run(args_sgd.build())
        
        # run sgd_dec*
        args_sgd_dec = copy.deepcopy(defaults_sgd_dec)
        args_sgd_dec.gpu(gpu.next())
        args_sgd_dec.lr(0.0728).momentum(0.16).regularization(0.07181).lr_decay(0.9970)
        r.run(args_sgd_dec.build())
    
        # run tgdm 72
        args_tgdm = copy.deepcopy(defaults_tgdm)
        args_tgdm.gpu(gpu.next())    
        args_tgdm.outer_iters(1).inner_iters(72)
        r.run(args_tgdm.build())
        
        # run tgdm 39
        args_tgdm = copy.deepcopy(defaults_tgdm)
        args_tgdm.gpu(gpu.next())
        args_tgdm.outer_iters(1).inner_iters(39)
        r.run(args_tgdm.build())
        
        # run tgdm 5
        args_tgdm = copy.deepcopy(defaults_tgdm)
        args_tgdm.gpu(gpu.next())
        args_tgdm.hlr_lr(0.0001).hlr_momentum(0.0001).hlr_regularization(0.0001)    
        args_tgdm.outer_iters(1).inner_iters(5)
        r.run(args_tgdm.build())
        
        # run hd
        #args_hd = copy.deepcopy(defaults_hd)
        #args_hd.gpu(gpu.next())    
        #r.run(args_hd.build())
        
        # run hdc
        args_hdc = copy.deepcopy(defaults_hdc)
        args_hdc.gpu(gpu.next())    
        r.run(args_hdc.build())
        
        # run hdc*5
        #args_hdc = copy.deepcopy(defaults_hdc)
        #args_hdc.gpu(gpu.next()) 
        #args_hdc.hlr_lr(0.005).hlr_momentum(0.005).hlr_regularization(0.0001)    
        #r.run(args_hdc.build())
        
        # run t1t2    
        args_t1t2 = copy.deepcopy(defaults_t1t2)
        args_t1t2.gpu(gpu.next())    
        r.run(args_t1t2.build())
        
        # run sgd
        args_sgd = copy.deepcopy(defaults_sgd)
        args_sgd.gpu(gpu.next())        
        r.run(args_sgd.build())
        
        # run sgd_dec
        args_sgd_dec = copy.deepcopy(defaults_sgd_dec)
        args_sgd_dec.gpu(gpu.next())        
        r.run(args_sgd_dec.build())    
    
    # wait for it
    r.wait() 

def cifar10_tgdm_bufix(factory):
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(15000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512')
    # CIFAR10
    defaults.dataset(Dataset.CIFAR10).batch_size(64).train_split(0.1).valid_split(0.11)
    #defaults.dataset(Dataset.CIFAR10).batch_size(64).train_split(0.50).valid_split(0.55)
        
    # optimizer
    #defaults.optimizer(Optimizer.TGDM)
    #defaults.optimizer(Optimizer.TGDM_T1T2)    
    # settings
    
    #tgdm
    defaults.lr(0.03).momentum(0.5).regularization(0.1)        

    # densenet:
    #defaults.hlr_lr(0.0001).hlr_momentum(0.001).hlr_regularization(0.01) # beta 0.7-0.9
    
    # to test (simple updates)
    #defaults.validation_iteration(100) # fix modulo 0
    #inner_iters = [5, 1]
    #outer_iters = [1]    
    
    # crazy updates:
    defaults.validation_iteration(100) # fix modulo 0
    inner_iters = [39, 39]    
    outer_iters = [1]
    defaults.hlr_lr(0.001).hlr_momentum(0.001).hlr_regularization(0.001/10) #
    
    # add DA
    defaults.data_augmentation(True)
    
    # regulation
    regulations = ['L2']
    optimizers = [Optimizer.TGDM]    
    #optimizers = [Optimizer.TORCH_SGD]    
    #defaults.lr_decay_iterations(3000)   
    
    for iters in inner_iters:
        for outers in outer_iters:
            for regulation in regulations:
                for optimizer in optimizers:
                    args = copy.deepcopy(defaults)
                    args.gpu(gpu.next())
                    args.inner_iters(iters)
                    args.outer_iters(outers)
                    args.regulation(regulation)
                    args.optimizer(optimizer)
                    
                    # add validation to training for hd methods:
                    #if optimizer in [Optimizer.TGDM_HD, Optimizer.TGDM_HDC]:
                    #    print('add validation as training data! (HD-Variants)')
                    #    args.train_split(args.args['--valid_split'])
                    #    args.valid_split(args.args['--valid_split'] + 0.1)
                    
                    r.run(args.build())
    r.wait()  
    
def birds_tgdm(factory):
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(15000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')    
    
    # BIRDS:
    defaults.dataset(Dataset.BIRDS).batch_size(128).train_split(0.9).valid_split(1.0)
    
    # settings
    
    #tgdm    
    defaults.lr(0.01).momentum(0.5).regularization(0.0001)    

    # densenet:
    #defaults.hlr_lr(0.0001).hlr_momentum(0.001).hlr_regularization(0.01) # beta 0.7-0.9
    
    # crazy updates:
    defaults.validation_iteration(100) # fix modulo 0
    inner_iters = [22, 22]
    outer_iters = [1]
    defaults.hlr_lr(0.001).hlr_momentum(0.001).hlr_regularization(0.001/10) #
    
    # add DA
    #defaults.data_augmentation(True) # birds no da
    
    # regulation
    regulations = ['L2']
    #optimizers = [Optimizer.TGDM]    
    optimizers = [Optimizer.TORCH_SGD]    
    defaults.lr_decay_iterations(400)   
    defaults.lr(0.1).momentum(0.5).regularization(0.0001)    
    
    for iters in inner_iters:
        for outers in outer_iters:
            for regulation in regulations:
                for optimizer in optimizers:
                    args = copy.deepcopy(defaults)
                    args.gpu(gpu.next())
                    args.inner_iters(iters)
                    args.outer_iters(outers)
                    args.regulation(regulation)
                    args.optimizer(optimizer)
                    
                    # add validation to training for hd methods:
                    #if optimizer in [Optimizer.TGDM_HD, Optimizer.TGDM_HDC]:
                    #    print('add validation as training data! (HD-Variants)')
                    #    args.train_split(args.args['--valid_split'])
                    #    args.valid_split(args.args['--valid_split'] + 0.1)
                    
                    r.run(args.build())
    r.wait() 

def scenario_b_random_search(factory):
    ''' runs a large random search and tries to converge '''
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(5000).group(name())
    
    # unknown settings
    defaults.architecture('densenet121')
    defaults.dataset(Dataset.CIFAR100)
    defaults.batch_size(256).train_split(0.9).valid_split(1.0)
    defaults.validation_iteration(400)
    
    # tgdm
    defaults_tgdm = copy.deepcopy(defaults)
    defaults_tgdm.inner_iters(5).optimizer(Optimizer.TGDM).regulation('L2')
    defaults_tgdm.hlr_lr(0.0003).hlr_momentum(0.0002).hlr_regularization(0.00003)
    
    # sgd
    defaults_sgd = copy.deepcopy(defaults)    
    defaults_sgd.optimizer(Optimizer.TORCH_SGD).regulation('L2')
    defaults_sgd.lr_decay_iterations(1000)
    
    # run random search:
    for i in range(100):
        lr = RandomUniformValue(0.0001, 1.0, ExpValue()).get_value().item()
        momentum = RandomUniformValue(0.1, 0.9, SigmoidValue()).get_value().item()
        regularization = RandomUniformValue(0.0001, 1.0, ExpValue()).get_value().item()
        print('use random entry: ', lr, momentum, regularization)
        
        # run tgdm
        args_tgdm = copy.deepcopy(defaults_tgdm)
        args_tgdm.gpu(gpu.next())
        args_tgdm.lr(lr).momentum(momentum).regularization(regularization)
        r.run(args_tgdm.build())
        
        # run sgd
        args_sgd = copy.deepcopy(defaults_sgd)
        args_sgd.gpu(gpu.next())
        args_sgd.lr(lr).momentum(momentum).regularization(regularization)
        r.run(args_sgd.build())
    
    # wait for it
    r.wait()    

def scenario_a_reuse(factory):
    ''' runs sgd and tgdm, reusing their 10% hyperparameters on 50% of the data '''
    
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(36000).group(name())
    
    # build defaults
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512') 
    
    # dataset
    defaults.dataset(Dataset.CIFAR10)    
    defaults.batch_size(64).train_split(0.5).valid_split(0.55) # use 50%
    defaults.validation_iteration(1000)
    
    # data augmentation:
    defaults.data_augmentation(True)
    
    # tgdm
    args = copy.deepcopy(defaults)
    args.gpu(gpu.next())
    args.inner_iters(5).optimizer(Optimizer.TGDM).regulation('L2')
    args.lr(0.003).momentum(0.2).regularization(0.1)
    defaults.hlr_lr(0.0001).hlr_momentum(0.0001).hlr_regularization(0.0001/10)
    r.run(args.build())
    
    #sgd
    args = copy.deepcopy(defaults)
    args.gpu(gpu.next())
    args.optimizer(Optimizer.TORCH_SGD).regulation('L2')
    args.lr(0.003).momentum(0.4).regularization(0.1).lr_decay_iterations(3000) # extend training time
    r.run(args.build())
    
    # wait for results
    r.wait()    

def find_lr_reg_momentum(factory):
    ''' Does a grid search in order to find best initial lr, momentum, regularization '''

    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(12000).group(name())
    
    # network
    #defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    defaults.architecture('convnet512') 
    
    # dataset
    defaults.dataset(Dataset.CIFAR10)    
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)
    
    # optimizer
    #defaults.inner_iters(5).optimizer(Optimizer.TGDM).regulation('L2')
    defaults.optimizer(Optimizer.TORCH_SGD).regulation('L2').lr_decay_iterations(3000)
    
    # so far estimated params:
    #lr = 0.1
    #regularization = 0.0003
    #momentum = None
    #momentums = [0.4, 0.2]
    
    # SGD
    lr = 0.1
    regularization = 0.001
    #regularization = 0
    momentum = None
    momentums = [0.6, 0.2]
    
    
    # search values:
    lrs = [0.1, 0.05, 0.01, 0.005]    
    #regularizations = [0.2, 0.1, 0.05, 0.01]
    regularizations = [0.01, 0.005, 0.001, 0.0005]    
    #momentums = [0.8, 0.6, 0.4, 0.2]
    #momentums = [0.6, 0.2]
    
    # tune hlrs:
    hlr_lr = 0.0005/10
    hlr_regularization = 0.001/10
    hlr_momentum = 0.0001/2 
    
    # log of converged values:
    #### convnet-512, tgdm:
    # lr=0.1, regularization=0.0003, momentum=0.2-0.4
    # hlr_alpha 0.0005
    # hlr_regularization = 0.001
    # hlr_momentum = 0.00005
    #### convnet-512, sgd:
    # lr = 0.1, reg=0.001, momentum=0.2-0.6
    
    
    if lr is None:
        defaults.hlr_lr(hlr_lr).hlr_regularization(0).hlr_momentum(0)
        defaults.regularization(0).momentum(0.5)        
        find_lr(gpu, r, defaults, lrs)
        return
    if regularization is None:
        defaults.hlr_lr(hlr_lr).hlr_regularization(hlr_regularization).hlr_momentum(0)        
        defaults.lr(lr).momentum(0.5)        
        find_regularization(gpu, r, defaults, regularizations)
        return
    if momentum is None:
        defaults.hlr_lr(hlr_lr).hlr_regularization(hlr_regularization).hlr_momentum(hlr_momentum)        
        defaults.lr(lr).regularization(regularization)       
        find_momentum(gpu, r, defaults, momentums)
        return

def find_lr(gpu, r, defaults, lrs):
    # build grid search:    
    for lr in lrs:
        args = copy.deepcopy(defaults)
        args.gpu(gpu.next())
        args.lr(lr)
        r.run(args.build()) # go!
    r.wait()
def find_regularization(gpu, r, defaults, regularizations):
    # build grid search:        
    for regularization in regularizations:
        args = copy.deepcopy(defaults)
        args.gpu(gpu.next())
        args.regularization(regularization)
        r.run(args.build()) # go!
    r.wait()
def find_momentum(gpu, r, defaults, momentums):
    # build grid search:        
    for momentum in momentums:
        args = copy.deepcopy(defaults)
        args.gpu(gpu.next())
        args.momentum(momentum)
        r.run(args.build()) # go!
    r.wait()
    


def cifar10_tgdm_gs_lr(factory):
    ''' Does a grid search in order to find best initial lr, momentum, regularization '''
       
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(2000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512')
    
    # dataset
    defaults.dataset(Dataset.CIFAR10)    
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)
    
    # optimizer
    defaults.inner_iters(5).optimizer(Optimizer.TGDM).regulation('L2')    
    
    
    # default init
    defaults.momentum(0.5)
    
    # phase 1: find learning rate
    #hlr_lr = 0.0001
    #hlr_regularization = 0    
    #lrs = [0.1, 0.05, 0.01, 0.005]
    #regularizations = [0]
    
    # phase 2: find regularization:
    #hlr_lr = 0.0001
    #lrs = [0.03]
    #hlr_regularization = 0.001 # L2
    #hlr_regularization = 0.000001 # L1
    #hlr_regularization = 0.001 # noise    
    #regularizations = [0.2, 0.1, 0.05, 0.01] #L2
    #regularizations = [0.002, 0.001, 0.0005, 0.0001] #L1
    regularizations = [0.002, 0.001, 0.0005, 0.0001] #noise
    
    # phase 3: find beta:
    hlr_lr = 0.0001
    lrs = [0.03]
    hlr_regularization = 0.00001 # L2+beta  
    regularizations = [0.1]
    hlr_momentum = 0.0001
    momentums = [0.8, 0.6, 0.4, 0.2]
    
    # set hlrs:
    defaults.hlr_lr(hlr_lr).hlr_momentum(hlr_momentum).hlr_regularization(hlr_regularization)
   
    # build grid search:    
    for lr in lrs:
        for regularization in regularizations:
            for momentum in momentums:
                args = copy.deepcopy(defaults)
                args.gpu(gpu.next())
                args.lr(lr).regularization(regularization).momentum(momentum)
                r.run(args.build()) # go!
    r.wait()  

def cifar10_sgd_gs(factory):
    ''' Does a grid search with SGD and lr decays '''
    
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(2000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512')
    
    # dataset
    defaults.dataset(Dataset.CIFAR10)    
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)
    
    # optimizer
    defaults.inner_iters(5).optimizer(Optimizer.TORCH_SGD).regulation('L2')    
        
    # phase 1
    lr_decays = [1000, 500, 100]
    lr = 0.03
    lrs = [lr, lr*2, lr/2, lr*10, lr/10]    
    momentums = [0.8, 0.5, 0.2]
    regularization = 0.1
    regularizations = [regularization, regularization*5, regularization/5]
        
    # build grid search:  
    for lr_decay in lr_decays:
        for lr in lrs:
            for momentum in momentums:
                for regularization in regularizations:
                    args = copy.deepcopy(defaults)
                    args.gpu(gpu.next())                
                    args.lr(lr).momentum(momentum).regularization(regularization)
                    args.lr_decay_iterations(lr_decay) ## aaha b
                    r.run(args.build()) # go!
    r.wait() 
    

def cifar10_tgdm_hdc_gs_learningrates(factory):
    ''' Does a grid search in order to find best hyper learning rates '''
       
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(10000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512')
    
    # dataset
    defaults.dataset(Dataset.CIFAR10)    
    defaults.batch_size(64).train_split(0.1).valid_split(0.11)
    
    # optimizer
    defaults.inner_iters(5).optimizer(Optimizer.TGDM).regulation('L2')
    #defaults.optimizer(Optimizer.TORCH_SGD).regulation('L2')    
    
    # default init
    defaults.lr(0.03).momentum(0.4).regularization(0.1) # TGDM
    #defaults.lr(0.03).momentum(0.2).regularization(0.1) # SGD
    
    # phase 1
    hlr_lr = 0.0001
    #hlr_lrs = [0.0001, hlr_lr*2, hlr_lr/2, hlr_lr*10, hlr_lr/10]    
    hlr_lrs = [hlr_lr/2, hlr_lr/10]    
    hlr_regularization = 0.00015
    #hlr_regularizations = [hlr_regularization, hlr_regularization*5, hlr_regularization/5]
    hlr_regularizations = [hlr_regularization/5, hlr_regularization/10]
    
    # phase 2
    #hlr_lr = 0.0001
    #hlr_lrs = [hlr_lr*1.5, hlr_lr/1.5]
    #hlr_regularization = 0.003
    #hlr_regularizations = [hlr_regularization*2, hlr_regularization/2]
    
    # phase 3
    #hlr_lr = 0.0001
    #hlr_lrs = [hlr_lr, hlr_lr*1.2, hlr_lr/1.2]
    #hlr_regularization = 0.000015
    #hlr_regularizations = [hlr_regularization, hlr_regularization*1.5, hlr_regularization/1.5]
    
    # build grid search:    
    for hlr_lr in hlr_lrs:
        for hlr_regularization in hlr_regularizations:
            args = copy.deepcopy(defaults)
            args.gpu(gpu.next())
            args.hlr_lr(hlr_lr)
            args.hlr_momentum(hlr_lr)
            args.hlr_regularization(hlr_regularization)
            #args.hlr_lr(hlr_lr*5)
            #args.hlr_momentum(hlr_lr*15/3) # use same as lr, 66t1t2
            #args.hlr_regularization(hlr_regularization*15)
            r.run(args.build()) # go!
    r.wait()    
    
    

def cifar10_tgdm(factory):
    gpu, r = factory.create(name())
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(20000).group(name())
    
    # network
    defaults.architecture('resnet18')
    #defaults.architecture('densenet121')
    #defaults.architecture('convnet512')
    defaults.dataset(Dataset.CIFAR10).batch_size(64).train_split(0.1).valid_split(0.2)
    
    # optimizer
    #defaults.optimizer(Optimizer.TGDM)
    #defaults.optimizer(Optimizer.TGDM_T1T2)    
    
    
    # settings
    
    #tgdm
    defaults.lr(0.03).momentum(0.5).regularization(0.1)
    
    # resnet paper:
    #defaults.lr(0.03).momentum(0.9).regularization(0.00001)
    defaults.batch_size(64).train_split(0.9).valid_split(1.0)#.data_augmentation(True)
    
    
    #defaults.lr(0.03).momentum(0.5).regularization(0.01)
    #defaults.hlr_lr(0.00001/2/2).hlr_momentum(0.0001).hlr_regularization(0.001) # beta 0.7-0.9
    #defaults.hlr_lr(0.00001/10).hlr_momentum(0.0001).hlr_regularization(0.0005) # beta 0.7-0.9, L1
    #defaults.hlr_lr(0.00001/10).hlr_momentum(0.0001).hlr_regularization(0.001) # beta 0.7-0.9, noise    
    #defaults.hlr_lr(0.00001).hlr_momentum(0.0001*0.03).hlr_regularization(0.01*0.03*0.5) # beta 0.7-0.9, no scale
    
    # convex sgd ?
    defaults.hlr_lr(0.001/10).hlr_momentum(0.00001).hlr_regularization(0.000001) #
    
    #defaults.hlr_lr(0.0001).hlr_momentum(0.01*0.03/3).hlr_regularization(0.01*0.03*0.5) #works for 90 and 10%}
    
    
    
    # t1t2:
    #defaults.hlr_lr(0.00001/5.).hlr_momentum(0.0001/5.).hlr_regularization(0.01/5.) # beta 0.7-0.9
    
    # hd/hdc:
    #defaults.hlr_lr(0.00001/2).hlr_momentum(0.0001).hlr_regularization(0.01) # beta 0.7-0.9
    #aa
    
    
    # densenet:
    #defaults.hlr_lr(0.0001).hlr_momentum(0.001).hlr_regularization(0.01) # beta 0.7-0.9
    
    # to test
    defaults.validation_iteration(780) # fix modulo 0
    inner_iters = [78*2*9, 78*4*9]
    outer_iters = [9]    
    #regulations = ['L2', 'L1', 'noise']
    regulations = ['L2']
    #optimizers = [Optimizer.TGDM_HDC, Optimizer.TGDM_HDC]
    optimizers = [Optimizer.TGDM]
    #optimizers = [Optimizer.TGDM_T1T2, Optimizer.TGDM_T1T2]
    
    for iters in inner_iters:
        for outers in outer_iters:
            for regulation in regulations:
                for optimizer in optimizers:
                    args = copy.deepcopy(defaults)
                    args.gpu(gpu.next())
                    args.inner_iters(iters)
                    args.outer_iters(outers)
                    args.regulation(regulation)
                    args.optimizer(optimizer)
                    
                    # add validation to training for hd methods:
                    #if optimizer in [Optimizer.TGDM_HD, Optimizer.TGDM_HDC]:
                    #    print('add validation as training data! (HD-Variants)')
                    #    args.train_split(args.args['--valid_split'])
                    #    args.valid_split(args.args['--valid_split'] + 0.1)
                    
                    r.run(args.build())
    r.wait()        
 

if __name__ == '__main__':
    main()