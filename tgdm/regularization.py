
import torch

class RegularizationFactory():
    
    def by_name(self, name):
        if name == 'L2':
            return L2Regularization()
        if name == 'L1':
            return L1Regularization()
        if name == 'noise':
            return NoiseRegularization()
        raise ValueError()

class RegularizationStrategy():
    ''' Base class for our type of regularizers.
        The derivative of the \Omega function.    
    '''
    
    def step(self, w):
        ''' returns the step direction for inner optimization (without scaling by lambda)
        (and directly used for the hypergradient)
        '''
        pass


class L2Regularization(RegularizationStrategy):
    ''' L2 Regularization '''
    
    def step(self, w):
        return w
    
class L1Regularization(RegularizationStrategy):
    
    def step(self, w):
        return torch.sign(w)

class NoiseRegularization(RegularizationStrategy):
    
    def step(self, w):
        epsilon = torch.randn_like(w)
        return epsilon
    