import inspect
import copy

from tgdm.parametrized_value import RandomUniformValue, ExpValue, SigmoidValue, ParametrizedValue

from experiment import local_machine, ExperimentFactory, Optimizer, MartheExperimentArgBuilder
from dataset import Dataset

# say my name:
name = lambda : inspect.stack()[1][3]

def main():
    concurrent_tasks = 1 if local_machine() else 2
    factory = ExperimentFactory(concurrent_tasks, 'marthe_experiment.py')
        
    # experiments
    
    # marthe
    cifar10_marthe(factory)


def cifar10_marthe(factory):
    ''' runs a large random search and tries to converge '''
    gpu, r = factory.create(name())
    defaults = MartheExperimentArgBuilder()
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
    defaults_marthe = copy.deepcopy(defaults)
    defaults_marthe.optimizer('sgd') # l2 is default    
    
    #for mu in [-1., 0.99999]:
    #for mu in [0.9999, 0.999]:
    #for mu in [0.999999, 0.9999999]:
    for mu in [-1, -1]:
        args_marthe = copy.deepcopy(defaults_marthe)
        args_marthe.gpu(gpu.next())
        args_marthe.mu(mu)
        r.run(args_marthe.build())
    
    # wait for it
    r.wait() 


if __name__ == '__main__':
    main()