import numpy as np 
import random
import json
import os
import shutil
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Normalize, Resize, Compose, ToTensor, RandomCrop
import sys
from torch.utils.data import Dataset, DataLoader

class gazetrack_dataset(Dataset):
    def __init__(self, root, phase='train', size=(128, 128), transform=True, v=True):
        self.root = root
        self.rand_amt = 10
        if(phase=='test'):
            self.files = glob(root+"*.jpg")
        else:
            self.files = glob(root+"/images/*.jpg")
            
        self.phase = phase
        self.size = size
        self.aug = self.get_transforms(self.phase, self.size)
        self.transform = transform
        if(v):
            print("Num files for " + phase + " = " + str(len(self.files)))
        
    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        fname = self.files[idx]
        with open(self.files[idx].replace('.jpg','.json').replace('images', 'meta')) as f:
            meta = json.load(f)
        w, h = image.size
        screen_w, screen_h = meta['screen_w'], meta['screen_h']
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        
        kps = [meta['leye_x1']/w, meta['leye_y1']/h, meta['leye_x2']/w, meta['leye_y2']/h, 
               meta['reye_x1']/w, meta['reye_y1']/h, meta['reye_x2']/w, meta['reye_y2']/h]
        
        l_eye = image.crop((max(0, lx), max(0, ly), max(0, lx+lw), max(0, ly+lh)))
        r_eye = image.crop((max(0, rx), max(0, ry), max(0, rx+rw), max(0, ry+rh)))
        
        l_eye = l_eye.transpose(Image.FLIP_LEFT_RIGHT)
        
        kps = torch.tensor(kps).float()
        
        out = torch.tensor([meta['dot_xcam'], meta['dot_y_cam']]).float()
        
        l_eye = self.aug(l_eye)
        r_eye = self.aug(r_eye)
        
        return self.files[idx], l_eye, r_eye, kps, out, screen_w, screen_h
    
    def get_transforms(self, phase, size):
        list_transforms = []
        if(phase=="train"):
            list_transforms = [Resize((size[0]+10,size[1]+10)),
                               RandomCrop((size[0],size[1])),
                               ToTensor(),
                               Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02)),]
            
        else:
            list_transforms = [Resize((size[0],size[1])),
                               ToTensor(),
                               Normalize(mean=(0.3741, 0.4076, 0.5425), std=(0.02, 0.02, 0.02)),]
        
        list_trfms = Compose(list_transforms)
        return list_trfms
    
    def __len__(self):
        return len(self.files)   
