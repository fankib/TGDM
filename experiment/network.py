import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import torchvision

class Classifier(nn.Module):
    
    def __init__(self, n_classes, pretrained):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained=False
    
    def loss(self, data_iterator, device):
        data, target = next(data_iterator)
        data = data.to(device)
        target = target.to(device)

        # predict
        out = self(data)
        L = f.nll_loss(f.log_softmax(out, dim=1), target).view(1)
        return L

class ConvNetClassifier(Classifier):
    
    def __init__(self, architecture, n_classes):
        super().__init__(n_classes, False)
        kernel_size = (3,3)
        if architecture == 'convnet512':
            layers = [3, 32, 64, 128, 256, 512]
        elif architecture == 'convnet2048':
            layers = [3, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            raise ValueError('Invalid architecture: {}'.format(architecture))
        
        # build convolutions:
        modules = []
        for i in range(1, len(layers)):
            modules.append(nn.Conv2d(layers[i-1], layers[i], kernel_size))
            modules.append(nn.BatchNorm2d(layers[i]))
            modules.append(nn.ReLU())        
        self.conv = nn.Sequential(*modules)
        
        # classifier:
        self.fc = nn.Linear(layers[-1], n_classes)        
    
    def forward(self, x):        
        features = self.conv(x)        
        features = f.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.fc(features)
        

class DensetNetClassifier(Classifier):
    
    def __init__(self, architecture, n_classes, pretrained=False):
        super().__init__(n_classes, pretrained)        
        if pretrained:
            print('use pretrained network!')
        elif architecture == 'densenet121':
            self.densenet = torchvision.models.densenet121(pretrained=pretrained)
        elif architecture == 'densenet161':
            self.densenet = torchvision.models.densenet161(pretrained=pretrained)
        elif architecture == 'densenet269':
            self.densenet = torchvision.models.densenet169(pretrained=pretrained)
        elif architecture == 'densenet201':
            self.densenet = torchvision.models.densenet201(pretrained=pretrained)
        else:
            raise ValueError('Invalid architecture: {}'.format(architecture))
            
        n_inputs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(n_inputs, n_classes)    
    
    def forward(self, x):
        return self.densenet(x)

class ResnetClassifier(Classifier):
    
    def __init__(self, architecture, n_classes, pretrained=False):
        super().__init__(n_classes, pretrained)        
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
            raise ValueError('Invalid architecture: {}'.format(architecture))
            
        n_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_inputs, n_classes)    
    
    def forward(self, x):
        return self.resnet(x)