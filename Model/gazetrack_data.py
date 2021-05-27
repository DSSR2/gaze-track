import cv2
import json
import os
import shutil
from glob import glob
from torchvision.transforms import Normalize, Resize, Compose, ToTensor

from torch.utils.data import Dataset

class gazetrack_dataset(Dataset):
    def __init__(self, root, phase='train', size=(128, 128), transform=True):
        self.root = root
        if(phase=='test'):
            self.files = glob(root)
        else:
            self.files = glob(root+"/images/*.jpg")
            
        self.phase = phase
        self.size = size
        self.aug = self.get_transforms(self.phase, self.size)
        self.transform = transform
        
    def __getitem__(self, idx):
        image = cv2.imread(self.files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fname = self.files[idx]
        with open(self.files[idx].replace('.jpg','.json').replace('images', 'meta')) as f:
            meta = json.load(f)
            
        screen_w, screen_h = meta['screen_w'], meta['screen_h']
        lx, ly, lw, lh = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        rx, ry, rw, rh = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        
        l_eye = image[max(0, ly):max(0, ly+lh), max(0, lx):max(0, lx+lw)]
        r_eye = image[max(0, ry):max(0, ry+rh), max(0, rx):max(0, rx+rw)]
        l_eye = cv2.flip(l_eye, 1)
            
        kps = torch.tensor([lx, ly, (lx+lw), (ly+lh), rx, ry, (rx+rw), (ry+rh)]).float()
        
        out = torch.tensor([meta['dot_xcam'], meta['dot_ycam']]).float()
        
        l_eye = self.aug(image=l_eye)['image']
        r_eye = self.aug(image=r_eye)['image']
        
        return self.files[idx], l_eye, r_eye, kps, out, screen_w, screen_h
    
    def get_transforms(self, phase, size):
        list_transforms = []
        list_transforms.extend(
            [
                Resize(size[0]),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensor(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms
    
    def __len__(self):
        return len(self.files)
    

    
if __name__ == '__main__':
    test_dataset = gazetrack_dataset('../../dataset/')
    print(len(m))