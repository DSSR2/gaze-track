import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class gazetrack_model(nn.Module):
    def __init__(self):
        super(gazetrack_model, self).__init__()        
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