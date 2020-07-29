
import torch

from .tgdm_base import TGDMBase, Buffer

class TGDM(TGDMBase):
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # get params per group:
            lr = self.get_lr(group).item()
            momentum = self.get_momentum(group).item()
            regularization = self.get_regularization(group).item()
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data                
                
                param_state = self.state[p]
                
                # perform regularization:
                regulation_step = self.regulation.step(p.data)
                if regularization != 0:
                    d_p.add_(regularization, regulation_step)          
                
                # accumulate gradient lambda
                param_state[Buffer.partial_M_lambda].add_(momentum, regulation_step.clone().flatten().detach())
                param_state[Buffer.partial_W_lambda].add_(-lr, param_state[Buffer.partial_M_lambda].clone().detach())
                
                # accumulate gradient beta
                param_state[Buffer.partial_M_beta].add_(1.0, d_p.clone().flatten().detach())
                param_state[Buffer.partial_W_beta].add_(-lr, param_state[Buffer.partial_M_beta].clone().detach())
                
                # add momentum
                if Buffer.gd_momentum not in param_state:
                    buf = param_state[Buffer.gd_momentum] = torch.clone(d_p).detach()
                else:
                    buf = param_state[Buffer.gd_momentum]                
                buf.mul_(momentum).add_(d_p)                
                p.data.add_(-lr, buf)
                
                # accumulate gradient alpha
                param_state[Buffer.partial_W_alpha].add_(1.0, buf.clone().flatten().detach())

        return loss
        
    
        