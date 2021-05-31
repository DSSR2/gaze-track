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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from gazetrack_data import gazetrack_dataset
from gazetrack_model import gazetrack_model

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import argparse
parser = argparse.ArgumentParser(description='Train GazeTracker')
parser.add_argument('--dataset_dir', default='../../dataset/', help='Path to converted dataset')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.016, type=float, help='Number of epochs')
parser.add_argument('--save_dir', default='./models/', help='Path store checkpoints')
parser.add_argument('--csv_dir', default='./models/', help='Path to store loss values as csv')
parser.add_argument('--model_name', default='gazetrack', help='Name of the model. Will be used while saving checkpoints, loss etc.')
parser.add_argument('--gpu', default=0, type=int, help='To use GPU or not. 0 for CPU, 1 for GPU')
parser.add_argument('--workers', default=10, type=int, help='Numboer of workers')
parser.add_argument('--load_cpkt', default=None, help='Path to saved checkpoint')


def get_dataloader(path, phase, shuff=True, batch_size=256, num_workers=0):
    dataset = gazetrack_dataset(path, phase=phase)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=shuff,)
    return loader

class gazetrack_trainer(object):
    def __init__(self, model, ds_root, bs=256, epochs=20, gpu=0, workers=0, lr=0.016, name='model', save_path='./models/', csv_path='./'):
        self.batch_size = bs
        self.csv_path = csv_path
        self.save_path = save_path
        self.lr = lr
        self.name = name
        self.num_epochs = epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        
        if(gpu==1):
            print("Using GPU")
            self.device = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
            
        self.net = model
        self.best_val_acc = 0
        self.best_val_loss = 100
        self.best_train_loss = 100
        self.losses = {phase: [] for phase in self.phases}
        
        self.loss = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.64, verbose=True)
        
        self.net = self.net.to(self.device)
        self.dataloaders ={"train":get_dataloader(ds_root+'/train', 'train', batch_size=self.batch_size, num_workers=workers), 
                           "val":get_dataloader(ds_root+'/val', 'val', batch_size=self.batch_size, num_workers=workers)}
        print(len(self.dataloaders['train']))
        self.acc_scores = {phase: [] for phase in self.phases}
        
    def forward(self, leye, reye, kps, out):
        leye = leye.to(self.device)
        reye = reye.to(self.device)
        kps = kps.to(self.device)
        out = out.to(self.device)
        preds = self.net(leye, reye, kps)
        preds.to(self.device)
        loss = self.loss(preds, out)
        return loss

    def iterate(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0): 
            fname, leye, reye, kps, targets = batch[0], batch[1], batch[2], batch[3], batch[4]
            loss = self.forward(leye, reye, kps, targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            tmp = phase+'_loss'
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = running_loss/total_batches

        if(torch.cuda.is_available()):
            torch.cuda.empty_cache()
        return epoch_loss

    def train_end(self):
        train_loss = self.losses["train"]
        val_loss = self.losses["val"]
        df_data=np.array([train_loss, val_loss]).T
        df = pd.DataFrame(df_data,columns = ['train_loss', 'val_loss'])
        df.to_csv(self.csv_path+"/"+self.name+".csv")
        
    def predict(self):
        self.net.eval()
        with torch.no_grad():
            self.iterate(1,'test')
        print('Done')

    def fit(self):
        for epoch in range(0, self.num_epochs):
            train_loss = self.iterate(epoch, "train")
            self.losses["train"].append(train_loss)
            loss_p = np.nanmean(self.losses['train'])
            print(f'Train_Loss: {loss_p}')
            
            state = {
                "epoch": epoch,
                "best_loss": self.best_val_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if loss_p <  self.best_train_loss:
                print("* New train optimal found according to train loss, saving state *")
                torch.save(state, self.save_path+'/'+self.name+'_best_train.pth')
            self.net.eval()
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.losses["val"].append(val_loss)
                loss_p = np.nanmean(self.losses['val'])
                print(f'Validation_Loss: {loss_p}')
                self.scheduler.step()
            if val_loss <  self.best_val_loss:
                print("* New optimal found according to validation loss, saving state *")
                state["best_loss"] = self.best_val_loss = val_loss
                torch.save(state, self.save_path+'/'+self.name+'_'+str(val_loss)[:5]+'best_val_loss.pth')
            self.train_end()

if __name__ == '__main__':
    args = parser.parse_args()
    model = gazetrack_model()
                      
    trainer = gazetrack_trainer(model, ds_root=args.dataset_dir, bs=256, epochs=args.epochs, gpu=args.gpu, workers=args.workers, lr=args.lr, name=args.model_name, save_path=args.save_dir, csv_path=args.csv_dir)
                      
    if(args.load_cpkt):
        weights = torch.load(args.load_cpkt)['state_dict']
        trainer.net.load_state_dict(weights)
    print("Ready to train")
    trainer.fit()