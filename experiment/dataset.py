
import torch
import numpy as np
from torchvision import datasets, transforms

from dataset_birds import DatasetBirds

class Dataset():
    
    ''' Dataset abstraction to handle different train splits and data augmentation.
        supported datasets:
        * CIFAR10
        * CIFAR100
        * BIRDS
    '''
    
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    BIRDS = 'birds'   
    
    def by_name(name):
        name = name.lower()
        if name in [Dataset.CIFAR10, Dataset.CIFAR100, Dataset.BIRDS]:
            return name        
        raise ValueError('Unknown Dataset')        
    
    def __init__(self, name, batch_size, train_split, valid_split, data_augmentation):
        assert name in [Dataset.CIFAR10, Dataset.CIFAR100, Dataset.BIRDS]
        self.name = name
        self.batch_size = batch_size
        self.train_split = train_split # portion of training data for training
        self.valid_split = valid_split # portion of training data for validation (minus train_split)
        self.data_augmentation = data_augmentation
    
    def loaders(self):
        # init trainig dataset according to train_split:
        train_data = self.load_train_data()        
        train_len = len(train_data)
        indices = np.arange(train_len)
        np.random.shuffle(indices)
        ind_train_split = int(self.train_split*train_len)
        ind_valid_split = int(self.valid_split*train_len)
        train_indices = indices[:ind_train_split] # only use xx% of train data
        valid_indices = indices[ind_train_split:ind_valid_split] # use xx% of train data as validation
        train_subdata = torch.utils.data.Subset(train_data, train_indices)
        train_loader = torch.utils.data.DataLoader(train_subdata, batch_size=self.batch_size, shuffle=True)
        valid_subdata = torch.utils.data.Subset(train_data, valid_indices)
        valid_loader = torch.utils.data.DataLoader(valid_subdata, batch_size=self.batch_size, shuffle=True)
        
        # init test dataset (without data augmentation)
        test_data = self.load_test_data()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True) # shuffle for no patterns
        
        # log
        print('Use {} batches for training, {} batches for validation and {} batches for testing.'.format(int(len(train_subdata)/self.batch_size), int(len(valid_subdata)/self.batch_size), int(len(test_data)/self.batch_size)))
        
        return (train_loader, valid_loader, test_loader)
    
    def n_classes(self):
        if self.name == Dataset.CIFAR10:
            return 10
        if self.name == Dataset.CIFAR100:
            return 100
        if self.name == Dataset.BIRDS:
            return 200        
    
    def load_train_data(self):
        if self.name == Dataset.CIFAR10:
            return self.load_cifar10(True, self.train_transform())
        if self.name == Dataset.CIFAR100:
            return self.load_cifar100(True, self.train_transform())
        if self.name == Dataset.BIRDS:
            return self.load_birds(True, self.train_transform_birds())        
    
    def load_test_data(self):
        if self.name == Dataset.CIFAR10:
            return self.load_cifar10(False, self.test_transform())
        if self.name == Dataset.CIFAR100:
            return self.load_cifar100(False, self.test_transform())
        if self.name == Dataset.BIRDS:
            return self.load_birds(False, self.test_transform_birds())        
    
    def load_birds(self, train, transform):
        return DatasetBirds('~/tmp/birds', train=train, transform=transform)
    
    def train_transform_birds(self):
        if not self.data_augmentation:
            # imagenet:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            #return transforms.Compose([
                #transforms.Resize(32),
                #transforms.CenterCrop(32),
                #transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
                #])
        else:
            raise RuntimeError('not yet implemented')
    
    def test_transform_birds(self):
        #return transforms.Compose([
        #    transforms.Resize(32),
        #    transforms.CenterCrop(32),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
        #])
        # imagenet:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return 
    
    def load_cifar10(self, train, transform):
        return datasets.CIFAR10(root='~/tmp/cifar10', train=train, download=True, transform=transform)
    
    def load_cifar100(self, train, transform):
        return datasets.CIFAR100(root='~/tmp/cifar100', train=train, download=True, transform=transform)
    
    def train_transform(self):
        if not self.data_augmentation:
            return transforms.Compose([            
                #transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
            ])
        else:
            print('activate data augmentation!')
            return transforms.Compose([
                # simple data augmentation:
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),        
                transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),                
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
            ])
    
    def test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
        ])


        