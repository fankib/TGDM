
import torch

from .tgdm_base import TGDMBase, Buffer

class TGDM_HD(TGDMBase):
    ''' TGDM using the hyper descent (HD) approach
    '''
    
    ''' the two states:
        * L1 for processing the first batch
        * L2 for processing the second batch
    '''
    STATE_L1 = 'l1'
    STATE_L2 = 'l2'
    
    def __init__(self, params, defaults, regulation, logger):
        super().__init__(params, defaults, regulation, logger)
        self.hd_state = self.STATE_L1
    
    def zero_grad(self):
        if self.hd_state == self.STATE_L1:
            self.hyper_zero_grad()        
        super().zero_grad()     
    
    def hyper_zero_grad(self):
        self.reset_param_state_buffers([\
                    Buffer.partial_W_alpha,\
                    Buffer.partial_W_beta,\
                    Buffer.partial_W_lambda,\
                    Buffer.partial_E_w,\
                ])

    def step(self, closure=None):
        if self.hd_state == self.STATE_L1:
            loss = self.step_l1(closure)
            self.hd_state = self.STATE_L2
        else:
            loss = self.step_l2(closure)
            self.hyper_step()
            self.hd_state = self.STATE_L1
        return loss
    
    def step_l1(self, closure):
        ''' do the usual GDM step '''
        
        # ~~~ copy + paste: ~~~
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
                #param_state[Buffer.partial_W_lambda].add_(-lr*momentum, regulation_step.clone().flatten().detach())
                param_state[Buffer.partial_W_lambda].add_(-1.0, regulation_step.clone().flatten().detach())
                # accumulate gradient beta                
                #param_state[Buffer.partial_W_beta].add_(-lr, d_p.clone().flatten().detach())                                
                param_state[Buffer.partial_W_beta].add_(-1.0, d_p.clone().flatten().detach())
                # add momentum
                if Buffer.gd_momentum not in param_state:
                    buf = param_state[Buffer.gd_momentum] = torch.clone(d_p).detach()
                else:
                    buf = param_state[Buffer.gd_momentum]                
                buf.mul_(momentum).add_(d_p)                
                p.data.add_(-lr, buf)
                # accumulate gradient alpha
                param_state[Buffer.partial_W_alpha].add_(-1.0, buf.clone().flatten().detach())
        return loss
    
    def step_l2(self, closure):
        ''' do the GDM step but save the gradient for the hyperstep '''
                
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
                
                # accumulate gradient E (before or after regularization)
                param_state[Buffer.partial_E_w].add_(1.0, d_p.clone().flatten().detach())
                
                # add momentum
                if Buffer.gd_momentum not in param_state:
                    buf = param_state[Buffer.gd_momentum] = torch.clone(d_p).detach()
                else:
                    buf = param_state[Buffer.gd_momentum]                
                buf.mul_(momentum).add_(d_p)                
                p.data.add_(-lr, buf)
                
        return loss

    def hyper_step(self):
        ''' use the partial_E_w buffer instead of the params gradient  '''
        
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
                param_state = self.state[p]
                grad_E = param_state[Buffer.partial_E_w]
                
                # compute hypergradient alpha
                if use_lr:
                    alpha_grads.append((param_state[Buffer.partial_W_alpha]*grad_E).sum())
                
                if use_momentum:
                    beta_grads.append((param_state[Buffer.partial_W_beta]*grad_E).sum())
                
                if use_regularization:
                    lambda_grads.append((param_state[Buffer.partial_W_lambda]*grad_E).sum())
            
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


class TGDM_HDC(TGDM_HD):
    ''' TGDM using the hyper descent (HD) approach
        * does not use the regularization term of the second loss!
    '''
    
    def step_l2(self, closure):
        ''' do the GDM step but save the gradient for the hyperstep '''
        
        # ~~~ copy + paste ~~~        
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
                
                # accumulate gradient E (before regularization => HDC)
                param_state[Buffer.partial_E_w].add_(1.0, d_p.clone().flatten().detach())
                
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
                
        return loss