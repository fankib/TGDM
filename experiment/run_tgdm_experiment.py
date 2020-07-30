
import copy

from experiment import local_machine, ExperimentFactory, ArgBuilder
from dataset import Dataset

def main():
    concurrent_tasks = 1 if local_machine() else 2
    factory = ExperimentFactory(concurrent_tasks, 'tgdm_experiment.py')
    
    # experiments
    cifar10_tgdm(factory)

def cifar10_tgdm(factory):
    gpu, e = factory.create('cls18-tgdm-debug')
    defaults = TGDMExperimentArgBuilder()
    defaults.iterations(10000).group('cls18-tgdm-debug')
    
    # network
    defaults.architecture('resnet18')
    defaults.dataset(Dataset.CIFAR10).train_split(0.1).valid_split(0.2)
    
    # to test
    inner_iters = [1, 5]
    
    for iters in inner_iters:
        args = copy.copy(defaults)
        args.gpu(gpu.next())
        args.inner_iters(iters)
        e.run(args.build())
    e.wait()        
    

class TGDMExperimentArgBuilder(ArgBuilder):
    
    def batch_size(self, arg):
        self.args.append('--batch_size {}'.format(arg))
        return self
    
    def iterations(self, arg):
        self.args.append('--iterations {}'.format(arg))
        return self
    
    def inner_iters(self, arg):
        self.args.append('--inner_iters {}'.format(arg))
        return self
    
    def train_split(self, arg):
        self.args.append('--train_split {}'.format(arg))
        return self
    
    def valid_split(self, arg):
        self.args.append('--valid_split {}'.format(arg))
        return self
    
    def architecture(self, arg):
        self.args.append('--architecture {}'.format(arg))
        return self
    
    def optimizer(self, arg):
        self.args.append('--optimizer {}'.format(arg))
        return self
    
    def dataset(self, arg):
        self.args.append('--dataset {}'.format(arg))
        return self

    def data_augmentation(self, arg):
        self.args.append('--data_augmentation {}'.format(arg))
        return self

    def regulation(self, arg):
        self.args.append('--regulation {}'.format(arg))
        return self
    
    def hlr_lr(self, arg):
        self.args.append('--hlr_lr {}'.format(arg))
        return self

    def hlr_momentum(self, arg):
        self.args.append('--hlr_momentum {}'.format(arg))
        return self

    def hlr_regularization(self, arg):
        self.args.append('--hlr_regularization {}'.format(arg))
        return self

    def lr(self, arg):
        self.args.append('--lr {}'.format(arg))
        return self
    
    def momentum(self, arg):
        self.args.append('--momentum {}'.format(arg))
        return self
    
    def regularization(self, arg):
        self.args.append('--regularization {}'.format(arg))
        return self
 

if __name__ == '__main__':
    main()