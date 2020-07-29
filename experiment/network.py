import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import torchvision

''' classifier evaluation '''
def evaluate(model, loader, device):    
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)            
            out = model(data) # use test model with A*            
            losses.append(f.nll_loss(f.log_softmax(out, dim=1), target).item())
            pred = out.argmax(dim=1)            
            correct += pred.eq(target).sum().item()
            total += data.shape[0]
    return total, correct, np.mean(losses)

class ResnetClassifier(nn.Module):
    
    def __init__(self, architecture, n_classes, pretrained=False):
        super().__init__()        
        if pretrained:
            print('use pretrained network!')
        elif architecture == 'resnet18':
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif architecture == 'resnet34':
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif architecture == 'resnet50':
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif architecture == 'resnet101':
            self.resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif architecture == 'resnet152':
            self.resnet = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Invalid architecture: {}".format(architecture))
            
        n_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_inputs, n_classes)
    
    def loss(self, train_iterator, device):
        data, target = next(train_iterator)
        data = data.to(device)
        target = target.to(device)

        # predict
        out = self(data)
        L = f.nll_loss(f.log_softmax(out, dim=1), target).view(1)
        return L
    
    def forward(self, x):
        return self.resnet(x)