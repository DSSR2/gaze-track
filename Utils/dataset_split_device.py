import math, shutil, os, time, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image
import shutil
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Process

'''
Split converted dataset based on device for further experimentation and fine tuning. 
'''


parser = argparse.ArgumentParser(description='Split Dataset based on device')
parser.add_argument('--dataset_path', help="Path to dataset converted using prepareDataset.py. Should have train, test, val folders.")
parser.add_argument('--output_path', default=None, help="Path to store output dataset.")
parser.add_argument('--threads', default=1, type=int, help="Number of threads to process dataset")

devices = ['iPhone4S', 'iPhone5', 'iPhone5C', 'iPhone5S', 'iPhone6', 'iPhone6s', 'iPhone6Plus', 'iPhone6sPlus']

def convert_dataset(files, out_root):
    for i in tqdm(files): 
        with open(i) as f:
            data = json.load(f)
            device = data['device'].replace(' ', '')
        
        ds = i.split('/')[-3]
        img_file = i.replace('meta', 'images').replace('.json', '.jpg')
        out_path = out_root+'/'+device+"/"+ds+"/"
        
        shutil.copy(img_file, out_path+"/images/")
        shutil.copy(i, out_path+"/meta/")

def add_ttv(path):
    os.mkdir(path+"/train/")
    os.mkdir(path+"/val/")
    os.mkdir(path+"/test/")
    os.mkdir(path+"/train/images/")
    os.mkdir(path+"/train/meta/")
    os.mkdir(path+"/val/images/")
    os.mkdir(path+"/val/meta/")
    os.mkdir(path+"/test/images")
    os.mkdir(path+"/test/meta")
    
def preparePath(path, clear=True):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)
    for i in devices:
        os.mkdir(path+"/"+i+"/")
        add_ttv(path+"/"+i+"/")
    return path

def main():
    args = parser.parse_args()
    threads = args.threads
    preparePath(args.output_path)
    
    procs = []
    files = glob(args.dataset_path+"/*/meta/*.json")
    chunk = len(files)//threads
    print("Number of files: ", len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=convert_dataset, args=(f, args.output_path))
        procs.append(proc)
        proc.start()
        print(i)
        
    for proc in procs:
        proc.join()
        
    print("DONE")
    return 0

if __name__ == '__main__':
    main()