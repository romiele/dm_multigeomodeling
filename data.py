import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms
import PIL
import os
import numpy as np

class DiffSet(Dataset):
    def __init__(self, train, dataset_name):

        ds_mapping = {
            "MNIST": (MNIST, 32, 1),
            "FashionMNIST": (FashionMNIST, 32, 1),
            "CIFAR10": (CIFAR10, 32, 3),
        }

        t = transforms.Compose([transforms.ToTensor()])
        ds, img_size, channels = ds_mapping[dataset_name]
        ds = ds("./data", download=True, train=train, transform=t)

        self.ds = ds
        self.dataset_name = dataset_name
        self.size = img_size
        self.depth = channels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        ds_item = self.ds[item][0]

        if self.dataset_name == "MNIST" or self.dataset_name == "FashionMNIST":
            pad = transforms.Pad(2)
            data = pad(ds_item) # Pad to make it 32x32
        else:
            data = ds_item
        
        data = (data * 2.0) - 1.0 # normalize to [-1, 1].
        return data
    
    
class FaciesSet(Dataset):
    
    def __init__(self,dataset_name,root_dir=None, multi=False):
        
        # if 'syn' in dataset_name:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomInvert(1),
            transforms.RandomVerticalFlip(1),
            # transforms.RandomHorizontalFlip(),
            ])
        # else: 
        #     t= transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.RandomHorizontalFlip(),
        #         #torchvision.transforms.RandomVerticalFlip()     
        #         ])
        
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        
        self.size = t(PIL.Image.open(self.root_dir+'/'+self.dataset_name+f'/Facies/0.png')).shape
        self.channels = 1
        self.len_data = len(os.listdir(self.root_dir+'/'+self.dataset_name+'/Facies/'))
        self.transform = t
        self.multi= multi
        if self.multi:
            self.max_ip_reals=0 ; i=0; t = True
            while t==True:
                t = os.path.isfile(self.root_dir+'/'+self.dataset_name+f'/Ip/0_{i}.pt')
                if t==True: self.max_ip_reals +=1
                i+=1 
            self.ipmin = torch.load(self.root_dir+'/'+self.dataset_name+f'/Ip/10_8.pt', weights_only=True).min()
            self.ipmax = torch.load(self.root_dir+'/'+self.dataset_name+f'/Ip/10_8.pt', weights_only=True).max()
            #print(self.ipmin, self.ipmax)
            
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        facies_name = self.root_dir+'/'+self.dataset_name+f'/Facies/{idx}.png'
            
        facies = self.transform(PIL.Image.open(facies_name))
        
        #random sample a reandom 64x64 block in the 80x100 images
        if facies.shape[-1]==100:
            
            z= np.random.randint(0,80-64) 
            x= np.random.randint(0,100-64)
            facies = facies[0,None,z:z+64,x:x+64]
        
        else: 
            facies = self.transform(PIL.Image.open(facies_name))[0,None,:]
        
        # facies= facies*2-1
        
        if self.multi:
            Ip = self.root_dir + '/' + self.dataset_name+f'/Ip/{idx}_{np.random.randint(0,self.max_ip_reals)}.pt'
            Ip = torch.load(Ip, weights_only=True)
            Ip = Ip[0,None,z:z+64,x:x+64]
            Ip = (Ip - self.ipmin) / (self.ipmax - self.ipmin)
            #print(Ip)
            return torch.cat((facies,Ip), dim=0)
        
        else:
            
            return facies.detach()
