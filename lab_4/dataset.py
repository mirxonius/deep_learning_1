from typing import Iterable
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision

from pathlib import Path 

import torch
import matplotlib.pyplot as plt



class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train',remove_class = None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))
    
        if remove_class is not None:
            if isinstance(remove_class,Iterable):
                self.classes = [i for i in range(10) if i not in remove_class]
                idx = torch.ones(len(self.targets),dtype = bool)
                for i in range(len(self.targets)):
                    if self.targets[i] in remove_class:
                        idx[i] == False
                self.targets = self.targets[idx]
                self.images = self.images[idx] 
            

            if isinstance(remove_class,int):
                self.classes.remove(remove_class)
                idx = torch.ones(len(self.targets),dtype = bool)
                idx[self.targets == remove_class] = False
                self.targets = self.targets[idx]
                self.images = self.images[idx]


        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]




    def _sample_negative(self, index):
        # YOUR CODE HERE
        c = self.targets[index].item()
        neg = [cls for cls in self.classes if cls != c]
        neg = choice(neg)
        return choice(self.target2indices[neg])



    def _sample_positive(self, index):
        # YOUR CODE HERE
        c = self.targets[index].item()
        
        return choice(self.target2indices[c])

    
    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)




class CIFARMetricDataset(Dataset):

    def __init__(self,root="/tmp/cifar/", split='train',remove_class = None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        cifar_ds = torchvision.datasets.CIFAR10(self.root, train='train' in split, download=True)
        self.images, self.targets = cifar_ds.data / 255., cifar_ds.targets
        self.classes = list(range(10))
        self.images = torch.tensor(self.images,requires_grad = False,dtype = torch.float)
        self.images = torch.swapaxes(self.images,1,3)
        self.targets = torch.tensor(self.targets,requires_grad = False)        

        
        if remove_class is not None:
            if isinstance(remove_class,Iterable):
                self.classes = [i for i in range(10) if i not in remove_class]
                idx = torch.ones(len(self.targets),dtype = bool)
                for i in range(len(self.targets)):
                    if self.targets[i] in remove_class:
                        idx[i] == False
                self.targets = self.targets[idx]
                self.images = self.images[idx]  
            

            if isinstance(remove_class,int):
                self.classes.remove(remove_class)
                idx = torch.ones(len(self.targets),dtype = bool)
                idx[self.targets == remove_class] = False
                self.targets = self.targets[idx]
                self.images = self.images[idx]


        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]




    def _sample_negative(self, index):
        # YOUR CODE HERE
        c = self.targets[index].item()
        neg = [cls for cls in self.classes if cls != c]
        neg = choice(neg)
        return choice(self.target2indices[neg])



    def _sample_positive(self, index):
        # YOUR CODE HERE
        c = self.targets[index].item()
        
        return choice(self.target2indices[c])

    
    def __getitem__(self, index):
        anchor = self.images[index]
        target_id = self.targets[index].item()
        
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive, negative, target_id

    def __len__(self):
        return len(self.images)



if __name__ == "__main__":
    ds = MNISTMetricDataset(remove_class=[0,2])
    cif = CIFARMetricDataset()
    #Za indeks slike u datasetu trebam pronaći indeks neke druge iz iste klase
    #I pronaći inedks neke druge klase 

    #print(ds.target2indices[0])
    a,p,n,id = cif[0]
    print(a.shape)
    print(ds.classes)
    plt.imshow(a.view(32,32,3))
    plt.show()

    plt.imshow(p.view(32,32,3))
    plt.show()
    
    plt.imshow(n.view(32,32,3))
    plt.show()
