from comet_ml import Experiment

import cv2
import json
import os, time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_model import lit_gazetrack_model


import argparse
parser = argparse.ArgumentParser(description='Train GazeTracker')
parser.add_argument('--dataset_dir', default='../../dataset/', help='Path to converted dataset')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--save_dir', default='./models/', help='Path store checkpoints')
parser.add_argument('--gpus', default=1, type=int, help='Path store checkpoints')

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir)
    model = lit_gazetrack_model(args.dataset_dir, args.save_dir)
    trainer = pl.Trainer(gpus=args.gpus, accelerator="ddp", max_epochs=args.epochs, default_root_dir=args.save_dir, progress_bar_refresh_rate=1, auto_lr_find=True, auto_scale_batch_size='binsearch', callbacks=[checkpoint_callback])
    
    trainer.tune(model)
    print("DONE")
