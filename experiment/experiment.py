import subprocess
import shlex
import queue
import wandb
import numpy as np
import getpass
import time

import torch
import torch.nn.functional as f
from torchvision import transforms

def local_machine():
    return getpass.getuser() == 'fsb1'

class Optimizer():
    ''' Enumeration for the available optimizers
    '''    
    
    TGDM_HD = 'tgdm-hd'
    TGDM_HDC = 'tgdm-hdc'
    TGDM_T1T2 = 'tgdm-t1t2'    
    TGDM = 'tgdm'
    TORCH_SGD = 'torch-sgd'
    TORCH_ADAM = 'torch-adam'

class StopWatch():
    
    def __init__(self):
        self.ms_start = None
        self.ms_pause = None
    
    def start(self):
        self.ms_start = self.millis()
    
    def pause(self):
        assert self.ms_pause is None
        self.ms_pause = self.millis()
    
    def resume(self):
        assert self.ms_pause is not None
        ms_resume = self.millis()
        diff = ms_resume - self.ms_pause
        self.ms_start += diff
        self.ms_pause = None
    
    def current(self):
        ms_now = self.millis()
        if self.ms_pause is not None:
            ms_now = self.ms_pause                    
        return ms_now - self.ms_start
    
    def current_seconds(self):        
        millis = self.current()
        return millis/1000.
    
    def millis(self):
        return int(round(time.time() * 1000))        

class Maximum():
    
    def __init__(self):
        self.value = 0
        self.max_time = 0
    
    def test_and_set(self, value, time):
        if value > self.value:
            self.value = value
            self.max_time = time

class MinimumTimeAccuracy():
    ''' stores the time required to achieve a certain accuracy '''
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.time = -1
    
    def test_and_set(self, accuracy, time):
        if self.time == -1 and accuracy >= self.threshold:
            self.time = time
        

class ExperimentEvaluation():
    
    def __init__(self, logger):
        self.logger = logger        
        self.max_train_acc = Maximum()
        self.max_valid_acc = Maximum()
        self.max_test_acc = Maximum()
        self.time_test_acc_50 = MinimumTimeAccuracy(0.5)  
        self.time_test_acc_55 = MinimumTimeAccuracy(0.55)
        self.time_test_acc_60 = MinimumTimeAccuracy(0.6)
        
           
    def evaluate_loader(self, model, loader, device):    
        ''' classifier evaluation '''
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)            
                out = model(data)        
                losses.append(f.nll_loss(f.log_softmax(out, dim=1), target).item())
                pred = out.argmax(dim=1)            
                correct += pred.eq(target).sum().item()
                total += data.shape[0]
        return total, correct, np.mean(losses)
    
    def evaluate(self, model, iteration, train_loader, valid_loader, test_loader, time, device):        
        # train
        total, correct, avg_loss = self.evaluate_loader(model, train_loader, device)
        self.logger.log_evaluation('train', iteration, total, correct, avg_loss)
        self.max_train_acc.test_and_set(correct/total, time)        
        self.log_maximum('train', self.max_train_acc)
        # valid
        total, correct, avg_loss = self.evaluate_loader(model, valid_loader, device)
        self.logger.log_evaluation('valid', iteration, total, correct, avg_loss)
        self.max_valid_acc.test_and_set(correct/total, time)
        self.log_maximum('valid', self.max_valid_acc)
        # test
        total, correct, avg_loss = self.evaluate_loader(model, test_loader, device)
        test_acc = correct/total
        self.logger.log_evaluation('test', iteration, total, correct, avg_loss)    
        self.max_test_acc.test_and_set(test_acc, time)
        self.log_maximum('test', self.max_test_acc)
        self.time_test_acc_50.test_and_set(test_acc, time)
        self.time_test_acc_55.test_and_set(test_acc, time)
        self.time_test_acc_60.test_and_set(test_acc, time)
        self.logger.log({'time_test_acc_50': self.time_test_acc_50.time,\
                         'time_test_acc_55': self.time_test_acc_55.time,\
                         'time_test_acc_60': self.time_test_acc_60.time,\
                         })
        
    
    def log_maximum(self, prefix, maximum):
        self.logger.log({'{}_max_accuracy'.format(prefix): maximum.value,\
                         '{}_max_accuracy_time'.format(prefix): maximum.max_time})    

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
        runner = ScriptRunner(self.gpus, self.script, prefix)
        return gpu, runner

class ArgBuilder():
    ''' prepares command line arguments '''
    
    def __init__(self):
        self.args = {}
    
    def group(self, arg):
        self.args['--group'] = arg
        return self
    
    def gpu(self, arg):
        self.args['--gpu'] = arg
        return self
    
    def build(self):
        args = ['{} {}'.format(key, self.args[key]) for key in self.args]
        return ' '.join(args)

class TGDMExperimentArgBuilder(ArgBuilder):
    
    def batch_size(self, arg):
        self.args['--batch_size'] = arg
        return self
    
    def iterations(self, arg):
        self.args['--iterations'] = arg
        return self
    
    def inner_iters(self, arg):
        self.args['--inner_iters'] = arg
        return self
    
    def train_split(self, arg):
        self.args['--train_split'] = arg
        return self
    
    def valid_split(self, arg):
        self.args['--valid_split'] = arg
        return self
    
    def architecture(self, arg):
        self.args['--architecture'] = arg
        return self
    
    def optimizer(self, arg):
        self.args['--optimizer'] = arg
        return self
    
    def dataset(self, arg):
        self.args['--dataset'] = arg
        return self

    def data_augmentation(self, arg):
        self.args['--data_augmentation'] = arg
        return self

    def regulation(self, arg):
        self.args['--regulation'] = arg
        return self
    
    def hlr_lr(self, arg):
        self.args['--hlr_lr'] = arg
        return self

    def hlr_momentum(self, arg):
        self.args['--hlr_momentum'] = arg
        return self

    def hlr_regularization(self, arg):
        self.args['--hlr_regularization'] = arg
        return self

    def lr(self, arg):
        self.args['--lr'] = arg
        return self
    
    def momentum(self, arg):
        self.args['--momentum'] = arg
        return self
    
    def regularization(self, arg):
        self.args['--regularization'] = arg
        return self
    
    def lr_decay_iterations(self, arg):
        self.args['--lr_decay_iterations'] = arg
        return self
    
    def validation_iteration(self, arg):
        self.args['--validation_iteration'] = arg
        return self

class GpuDispatcher():
    ''' alternates gpus '''
    
    def __init__(self, gpus):        
        self.gpus = gpus
        self.counter = gpus-1
    
    def next(self):
        self.counter = (self.counter+1)%self.gpus
        return self.counter

class ScriptRunner():
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
                cmd = 'python -u {} {}'.format(self.script, args)                
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