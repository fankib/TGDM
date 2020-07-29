import os

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

class DatasetBirds(VisionDataset):
    
    ''' Implemements the VisionDataset as the DatasetFolder for CALTECH-BIRDS
        
        The Birds dataset requires additional crops in order to unify image size:
        * transforms.Resize(256)
        * transforms.CenterCrop(224) // or transforms.RandomResizedCrop(224)
    '''
    
    def __init__(self, root, train=True, loader=default_loader, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.train = train
        self.loader = loader
        self.loader_cache = {}
        
        self.samples = self.read_samples()
        if len(self.samples) == 0:
            raise RuntimeError('No samples found!')
        
    def read_samples(self):
        samples = []
        list_name = 'train.txt' if self.train else 'test.txt'
        list_path = 'lists/{}'.format(list_name)
        with open(os.path.join(self.root, list_path)) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            # join path
            img_path = os.path.join(self.root, 'images/{}'.format(line))
            # extract class:
            cls_str = line.split('.')[0]
            clsi = int(cls_str)-1 # 200 classes in range: 0-199
            samples.append((img_path, clsi))
        return samples
        
    def __getitem__(self, index):        
        path, target = self.samples[index]
        
        if path not in self.loader_cache:
            self.loader_cache[path] = self.loader(path)
        sample = self.loader_cache[path].copy()
            
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)

    def __len__(self):
        return len(self.samples)
