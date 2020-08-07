
import torch

from .tgdm_base import Buffer

class PYTORCH_SGD(torch.optim.Optimizer):
    
    def __init__(self, params, defaults, regulation, logger, lr_decay_iterations):        
        super().__init__(params, defaults)        
        assert regulation is not None    
        assert logger is not None        
        self.regulation = regulation
        self.logger = logger      
        self.lr_decay_iterations = lr_decay_iterations
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
                
            # update learning rate after this amount of steps:
            self.iterations += 1
            if self.iterations % self.lr_decay_iterations == 0:
                group['lr'] = lr/10
            
            # log hyper param values:
            self.logger.log({
                    'lr': group['lr'],\
                    'momentum': group['momentum'],\
                    'regularization': group['regularization'],\
                    })            
                

        return loss