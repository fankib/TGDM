
import torch

from .parametrized_value import ExpValue, SigmoidValue

class Buffer():
    ''' Buffers in use for each parameter group '''    
    # param_state buffers:
    partial_W_alpha = 'partial_W_alpha_buffer'    
    partial_W_beta = 'partial_W_beta_buffer'
    partial_M_beta = 'partial_M_beta_buffer'
    partial_W_lambda = 'partial_W_lambda_buffer'
    partial_M_lambda = 'partial_M_lambda_buffer'
    partial_E_w = 'partial_E_w_buffer' # for HD
    # group buffers:
    learning_rate = 'hyper_lr_buffer'
    momentum = 'hyper_momentum_buffer'
    regularization = 'hyper_regularization_buffer'
    gd_momentum = 'momentum_buffer'

class TGDMBase(torch.optim.Optimizer):
    ''' Base class for TGDM implementations
        There is not yet support for different param groups.
    '''
    
    def __init__(self, params, defaults, regulation, logger, inner_iters):        
        super().__init__(params, defaults)
        assert defaults['hyper_learning_rate'] is not None
        assert regulation is not None        
        assert logger is not None
        hyper_learning_rate = defaults['hyper_learning_rate']
        if '__len__' in dir(hyper_learning_rate):
            assert len(hyper_learning_rate) == 3, 'requires list of 3 hyper learning rates: \alpha, \beta, \lambda'
            self.hlr_lr = hyper_learning_rate[0]
            self.hlr_momentum = hyper_learning_rate[1]
            self.hlr_regularization = hyper_learning_rate[2]
        else:
            self.hlr_lr = self.hlr_momentum = self.hlr_regularization = hyper_learning_rate        
        
        self.regulation = regulation
        self.logger = logger
        self.inner_iters = inner_iters        
    
    def __setstate__(self, state):
        super().__setstate__(state)             
    
    def get_lr(self, group):
        if Buffer.learning_rate not in group:
            lr = torch.tensor(group['lr'])
            parameter_lr = group[Buffer.learning_rate] = ExpValue(lr)
        else:
            parameter_lr = group[Buffer.learning_rate]
        return parameter_lr.get_value()        
    
    def get_momentum(self, group):
        if Buffer.momentum not in group:
            momentum = torch.tensor(group['momentum'])
            parameter_momentum = group[Buffer.momentum] = SigmoidValue(momentum)
        else:
            parameter_momentum = group[Buffer.momentum]
        return parameter_momentum.get_value()        
    
    def get_regularization(self, group):
        if Buffer.regularization not in group:
            regularization = torch.tensor(group['regularization'])
            parameter_regularization = group[Buffer.regularization] = ExpValue(regularization)
        else:
            parameter_regularization = group[Buffer.regularization]
        return parameter_regularization.get_value()
    
    def reset_param_state_buffers(self, buffers):
        ''' inits the param state buffers with 0s '''
        for group in self.param_groups:            
            for p in group['params']:
                param_state = self.state[p]     
                for buffer in buffers:
                    param_state[buffer] = torch.zeros_like(p.flatten())                
                
    def hyper_zero_grad(self):
        pass
    
    def step(self, closure=None):
        pass
    
    def hyper_step(self):
        for group in self.param_groups:            
            
            use_lr = Buffer.learning_rate in group            
            use_momentum = Buffer.momentum in group
            use_regularization = Buffer.regularization in group                        
                        
            alpha_grads = []
            beta_grads = []
            lambda_grads = []
            alpha_grad = 0
            beta_grad = 0
            lambda_grad = 0
            
            for p in group['params']:                            
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                # compute hypergradient alpha
                if use_lr:
                    alpha_grads.append((param_state[Buffer.partial_W_alpha]*d_p.flatten()).sum())
                
                if use_momentum:
                    beta_grads.append((param_state[Buffer.partial_W_beta]*d_p.flatten()).sum())
                
                if use_regularization:
                    lambda_grads.append((param_state[Buffer.partial_W_lambda]*d_p.flatten()).sum())
            
            # apply hypergradients:
            if use_lr:
                parameter_lr = group[Buffer.learning_rate]
                alpha_grad = torch.tensor(alpha_grads).sum()
                parameter_lr.value.add_(-self.hlr_lr, self.hyper_momentum(group, 'alpha_grad_momentum', alpha_grad))
            
            if use_momentum:
                parameter_momentum = group[Buffer.momentum]
                beta_grad = torch.tensor(beta_grads).sum()
                parameter_momentum.value.add_(-self.hlr_momentum, self.hyper_momentum(group, 'beta_grad_momentum', beta_grad))
            
            if use_regularization:
                parameter_regularization = group[Buffer.regularization]
                lambda_grad = torch.tensor(lambda_grads).sum()
                parameter_regularization.value.add_(-self.hlr_regularization, self.hyper_momentum(group, 'lambda_grad_momentum', lambda_grad))
            
            # log hyper gradients
            self.logger.log({
                    'lambda_grad': lambda_grad,\
                    'alpha_grad': alpha_grad,\
                    'beta_grad': beta_grad})
            
            # log hyper param values:
            self.logger.log({
                    'lr': group[Buffer.learning_rate].get_value(),\
                    'momentum': group[Buffer.momentum].get_value(),\
                    'regularization': group[Buffer.regularization].get_value(),\
                    })
            
    def hyper_momentum(self, group, name, step, beta=0.5):
        if name not in group:
            buf = group[name] = step.clone().mul_(1.-beta)
            return buf        
        buf = group[name]
        buf.mul_(beta)
        buf.add_(1.-beta, step)
        return buf
    
    
        
    
    