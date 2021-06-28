import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from gazetrack_data import gazetrack_dataset
from torch.utils.data import DataLoader


class eye_model(nn.Module):
    def __init__(self):
        super(eye_model, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.02),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class landmark_model(nn.Module):
    def __init__(self):
        super(landmark_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class lit_gazetrack_model(pl.LightningModule):
    def __init__(self, data_path, save_path):
        super(lit_gazetrack_model, self).__init__()
        
        self.lr = 0.001
        self.batch_size = 1024
        self.data_path = data_path
        print("Data path: ", data_path)
        self.save_path = save_path
        
        self.eye_model = eye_model()
        self.lmModel = landmark_model()
        self.combined_model = nn.Sequential(nn.Linear(512+512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(4, 2),)
        
    

    def forward(self, leftEye, rightEye, lms):
        l_eye_feat = torch.flatten(self.eye_model(leftEye), 1)
        r_eye_feat = torch.flatten(self.eye_model(rightEye), 1)
        
        lm_feat = self.lmModel(lms)
        
        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat)
        return out
    
    def train_dataloader(self):
        train_dataset = gazetrack_dataset(self.data_path+"/train/", phase='train')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        dataVal = gazetrack_dataset(self.data_path+"/val/", phase='val')
        val_loader = DataLoader(dataVal, batch_size=self.batch_size, shuffle=True)
        return val_loader
    
    def training_step(self, batch, batch_idx):
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
        val_loss = F.mse_loss(y_hat, y)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        return optimizer
