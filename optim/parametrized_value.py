

class ParametrizedValue:
    
    def __init__(self, value):
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
        

class SigmoidValue(ParametrizedValue):
    ''' parametrized by v = sigmoid(a) (values between [0, 1])
    '''
    
    def set_value(self, value):
        self.value = (value / (1-value)).log()
    
    def get_value(self):
        return self.value.sigmoid()