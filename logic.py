import pandas as pd
import numpy as np
import torch, torchvision, os,pdb
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset

def collate_fn(batch):
    return tuple(zip(*batch))

class classifier(pl.LightningModule):
    def __init__(self,
                 model,
                 ds,
                 bs,
                 shuffle,
                 train_num_workers,
                 val_num_workers,
                 train_base_dir,
                 val_base_dir,
                 loss=torch.nn.CrossEntropyLoss(),
                 lr=2e-3,
                 wandb_run=None
                 ):
        
        super().__init__()
        self.train_imgs_paths = [os.path.join(train_base_dir,'images',p) for p in os.listdir(os.path.join(train_base_dir,'images'))]
        self.val_imgs_paths = [os.path.join(val_base_dir,'images',p) for p in os.listdir(os.path.join(val_base_dir,'images'))]
        self.train_df = pd.read_csv(os.path.join(train_base_dir,'train_ground_truth.txt'))
        self.val_df = pd.read_csv(os.path.join(val_base_dir,'val_ground_truth.txt'))
        self.bs = bs
        self.ds=ds
        self.shuffle = shuffle
        self.val_num_workers=val_num_workers
        self.train_num_workers = train_num_workers
        self.wandb_run = wandb_run
        self.model = model
        self.loss = loss
        self.lr = lr

    def val_dataloader(self) :
        val_ds = self.ds(self.val_imgs_paths,self.val_df)
        val_loader = DataLoader(val_ds,
                                batch_size=self.bs,
                                num_workers=self.val_num_workers,
                                collate_fn=collate_fn
                                )
        return val_loader
    
    def train_dataloader(self):
        train_ds = self.ds(self.train_imgs_paths,self.train_df)
        train_loader = DataLoader(train_ds,
                                  batch_size=self.bs,
                                  num_workers=self.train_num_workers,
                                  collate_fn=collate_fn
                                  )
        return train_loader
    
    def training_step(self, batch, batch_id ):
        imgs,targets = batch
        outputs = self.model(torch.stack(list(imgs), dim=0))
        loss_on_step = self.loss(outputs,torch.tensor(targets))
        #log on wandb
        if self.wandb_run:
            self.wandb_run.log({"train" : {"train_loss" : loss_on_step}}, commit = True)
        return loss_on_step
    
    def validation_step(self,batch,batch_id ):
        imgs,targets = batch
        outputs = self.model(torch.stack(list(imgs), dim=0))
        # pdb.set_trace()
        loss_on_step = self.loss(outputs,torch.tensor(targets))

        #log on wandb
        if self.wandb_run:
            self.wandb_run.log({"val" : {"val_loss" : loss_on_step}}, commit = False)
        return loss_on_step
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr)
