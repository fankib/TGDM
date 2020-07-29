import subprocess
import shlex
import queue
import wandb
import numpy as np
import getpass

import torch
from torchvision import transforms

def local_machine():
    return getpass.getuser() == 'fsb1'

class Logger():
    
    def __init__(self, project, group, log_wandb):
        self.project = project        
        self.current_step = 0        
        self.log_wandb = log_wandb
        self.group = group
        if log_wandb:
            wandb.init(project=project, group=group)
    
    def log_args(self, args):
        if self.log_wandb:
            wandb.config.update(args)
        print('~~~ args ~~~')
        args_d = args.__dict__
        for k in args_d.keys():
            print('{}: {}'.format(k, args_d[k]))
    
    def log_images(self, image_tensor, label="images", captions=None, normalize=False):        
        # denormalize using mu and sigma
        mu = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        sigma = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        imgs = []        
        for b in range(image_tensor.shape[0]):
            img = image_tensor[b].detach().cpu()*sigma + mu
            if normalize:
                img = (img - img.min()) / (img.max() - img.min())
            img = transforms.functional.to_pil_image(img)
            imgs.append(img)        
        if self.log_wandb:
            if captions is None:                
                wandb_imgs = [wandb.Image(img) for img in imgs]
            else:
                wandb_imgs = [wandb.Image(img, caption=c) for img,c in zip(imgs, captions)]            
            wandb.log({label: wandb_imgs}, step=self.current_step)            
        else:
            for i,img in enumerate(imgs):
                if captions is None:
                    img.save('imgs/{}-{}.png'.format(label, i))            
                else:
                    img.save('imgs/{}-{}-{}.png'.format(label, i, captions[i]))
    
    def step(self):
        self.current_step += 1
    
    def log(self, ord_dict):
        if self.log_wandb:
            wandb.log(ord_dict, step=self.current_step)        
        print(ord_dict)
    
    def log_list_values(self, prefix, values):
        self.log({'{}_mean'.format(prefix): np.mean(values),\
                  '{}_std'.format(prefix): np.std(values),\
                  '{}_min'.format(prefix): np.min(values),\
                  '{}_max'.format(prefix): np.max(values),\
                  '{}_argmin'.format(prefix): np.argmin(values),\
                  '{}_argmax'.format(prefix): np.argmax(values)})
    
    def log_evaluation(self, prefix, iteration, total, correct, avg_loss):
        acc = correct/total
        if self.log_wandb:
            wandb.log({'{}_accuracy'.format(prefix): acc,\
                       '{}_avg_loss'.format(prefix): avg_loss,\
                   'zz_{}_total'.format(prefix): total,\
                   'zz_{}_correct'.format(prefix): correct}, step=self.current_step)
        print('Iteration {} {} accuracy: {}/{}={:0.3f}'.format(iteration, prefix, correct, total, acc))

class ExperimentFactory():
    ''' creates GpuDispatcher and Evaluator '''
    
    def __init__(self, gpus, script):
        self.gpus = gpus
        self.script = script
    
    def create(self, prefix):
        gpu = GpuDispatcher(self.gpus)
        evaluator = Evaluator(self.gpus, self.script, prefix)
        return gpu, evaluator

class ArgBuilder():
    ''' prepares command line arguments '''
    
    def __init__(self):
        self.args = []
    
    def gpu(self, arg):
        self.args.append('--gpu {}'.format(arg))
        return self
    
    def build(self):
        return ' '.join(self.args)

class GpuDispatcher():
    ''' alternates gpus '''
    
    def __init__(self, gpus):        
        self.gpus = gpus
        self.counter = gpus-1
    
    def next(self):
        self.counter = (self.counter+1)%self.gpus
        return self.counter

class Evaluator():
    ''' little helper to execute python in parallel '''
    
    def __init__(self, jobs, script, prefix):        
        self.jobs = jobs
        self.script = script
        self.commands = queue.Queue()
        self.prefix = prefix            
    
    def run(self, args):  
        assert self.prefix is not None, 'Set prefix before calling run()'
        self.commands.put(args)                

    def wait(self):
        counter = 1        
        names = dict()
        logs = dict()
        while not self.commands.empty():
            
            # assume jobs take roughly the same amount of time
            
            # fill jobs:
            procs = queue.Queue()
            for i in range(self.jobs):
                args = self.commands.get()
                cmd = 'python -u {} --silent {}'.format(self.script, args)                
                cmds = shlex.split(cmd)
                print('run using {}'.format(args))
                p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
                procs.put(p)
                names[id(p)] = 'Task {}'.format(counter)
                logs[id(p)] = open('log/{}_task_{}'.format(self.prefix, counter), 'w')
                logs[id(p)].write('~~~ Run Experiment {} ~~~\n\n'.format(args))
                logs[id(p)].flush()
                counter += 1

            # process readlines:
            while not procs.empty():
                p = procs.get()                                  
                line = p.stdout.readline().decode('utf-8')
                logs[id(p)].write(line)
                logs[id(p)].flush()
                if line:
                    print('[{}]: {}'.format(names[id(p)], line.strip()))
                if p.poll() is None or line:
                    procs.put(p)  