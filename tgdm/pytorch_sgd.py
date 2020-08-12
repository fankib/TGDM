
import torch

from .tgdm_base import Buffer

class PYTORCH_SGD(torch.optim.Optimizer):
    
    def __init__(self, params, defaults, regulation, logger):        
        super().__init__(params, defaults)        
        assert regulation is not None    
        assert logger is not None        
        self.regulation = regulation
        self.logger = logger        
        self.iterations = 0
    
    def __setstate__(self, state):
        super().__setstate__(state)        
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # get params per group:
            lr = group['lr']
            momentum = group['momentum']
            regularization = group['regularization']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data                
                
                param_state = self.state[p]
                
                # perform regularization:
                regulation_step = self.regulation.step(p.data)
                if regularization != 0:
                    d_p.add_(regularization, regulation_step)                
                
                # add momentum
                if Buffer.gd_momentum not in param_state:
                    buf = param_state[Buffer.gd_momentum] = torch.clone(d_p).detach()
                else:
                    buf = param_state[Buffer.gd_momentum]                
                buf.mul_(momentum).add_(d_p)                
                p.data.add_(-lr, buf)
            
            # update lr
            self.iterations += 1
            self.update_lr(group)
            
            # log hyper param values:
            self.logger.log({
                    'lr': group['lr'],\
                    'momentum': group['momentum'],\
                    'regularization': group['regularization'],\
                    })            
        return loss
    
    def update_lr(self, group):
        pass

class PYTORCH_SGD_STEP(PYTORCH_SGD):
    ''' performs a step udate after the given amount of iterations '''
    
    def __init__(self, params, defaults, regulation, logger, lr_decay_iterations):
        super().__init__(params, defaults, regulation, logger)
        self.lr_decay_iterations = lr_decay_iterations
    
    def update_lr(self, group):
        # update learning rate after this amount of steps:        
        if self.iterations % self.lr_decay_iterations == 0:
            group['lr'] = group['lr']/10
    
class PYTORCH_SGD_DEC(PYTORCH_SGD):
    ''' performs exponential decay after each iteration '''
    
    def __init__(self, params, defaults, regulation, logger, lr_decay):
        super().__init__(params, defaults, regulation, logger)
        assert (lr_decay <= 1. and lr_decay > 0.)
        self.lr_decay = lr_decay
    
    def update_lr(self, group):
        group['lr'] = group['lr']*self.lr_decay
    