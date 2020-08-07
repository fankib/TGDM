import torch
import numpy as np

class ParametrizedValue:
    ''' parametrized value by v = a (no parametrization)
    '''
    
    def __init__(self, value=None):
        if value is not None:
            self.set_value(value)
    
    def set_value(self, value):
        self.value = value
    
    def get_value(self):
        return self.value


class ExpValue(ParametrizedValue):
    ''' parametrized by v = exp(a) (only positive values)
    '''
    
    def set_value(self, value):
        self.value = value.log()
    
    def get_value(self):
        return self.value.exp()

class Base10Value(ParametrizedValue):
    ''' parametrized by v = 10^a (only positive values)
    '''
    
    TEN = torch.tensor(10.0, requires_grad=False)
    
    def set_value(self, value):
        self.value = value.log10()
    
    def get_value(self):
        return self.TEN.pow(self.value)
    
        

class SigmoidValue(ParametrizedValue):
    ''' parametrized by v = sigmoid(a) (values between [0, 1])
    '''
    
    def set_value(self, value):
        self.value = (value / (1-value)).log()
    
    def get_value(self):
        return self.value.sigmoid()

class RandomUniformValue(ParametrizedValue):
    ''' Sets the value a (v = f(a)) uniform at random
    '''
    
    def __init__(self, low, high, parametrization):
        # lower bound:        
        parametrization.set_value(torch.tensor(low))
        a_low = parametrization.value
        # upper bound:
        parametrization.set_value(torch.tensor(high))
        a_high = parametrization.value
        # random value:
        value = torch.tensor(np.random.uniform(a_low.item(), a_high.item()))
        self.parametrization = parametrization
        self.parametrization.value = value
    
    def set_value(self, value):
        raise ValueError('you can not set values on randoms')
    
    def get_value(self):
        return self.parametrization.get_value()


    