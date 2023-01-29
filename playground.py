import wandb
import torch
import torchvision
import numpy as np
import pandas as pd

from pipeline import base_pipe
from logic import classifier
from model import get_model
import pytorch_lightning as pl

def main():
    wnadb_run = wandb.init(
                        project='Bravo5',
                        name='exp0'
                    )
    
    model = get_model(num_classes=7)


    Classifier = classifier(
                                model=model,
                                ds = base_pipe,
                                train_base_dir='counting/train',
                                val_base_dir='counting/val',
                                val_num_workers=4,
                                train_num_workers=4,
                                shuffle=True,
                                bs=64,
                                wandb_run= wnadb_run

    )

    Trainer=pl.Trainer(
                        # devices=1,
                        # accelerator="mps",
                        max_epochs=35,
                      )
    
    Trainer.fit(Classifier)
    return 

if __name__=="__main__":
    main()