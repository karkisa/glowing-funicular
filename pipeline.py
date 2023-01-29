import torch ,os
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pdb
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def conver_csv(path):
    df = pd.read_csv(path)
    print(df)

class base_pipe(Dataset):

    def __init__(   self,
                    paths,
                    df, 
                    shape = (90,90)) -> None:
        
        self.paths = paths
        self.df = df
        self.shape=shape
        super().__init__()

    def __len__(self):
        return len(self.paths)
    
    def read_img(self,path):
        img=torchvision.io.read_image(path)
        Resize = torchvision.transforms.Resize(self.shape)
        img = Resize(img)
        img = img/225.
        return img
    
    def get_label(self,path):
        name = path.split('/')[-1]
        label = self.df[self.df.Image==name]['count'].values[0]
        return label

    def __getitem__(self, index) :
        img = self.read_img(self.paths[index])
        label = self.get_label(self.paths[index])
        return img,label