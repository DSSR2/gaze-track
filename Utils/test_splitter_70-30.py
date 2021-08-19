import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split

'''
Split data according to 70/15/10 split based on unique points. 
This is the method Google follows to report the results obtained in their paper. 
'''
import argparse
parser = argparse.ArgumentParser(description='Split Dataset based on device')
parser.add_argument('--data_root', help="Path to a dataset converted using any of the dataset_converter_*.py files")
parser.add_argument('--out_root', help="Path to store the new split test set")
parser.add_argument('--threads', type=int, help="Number of workers")
parser.add_argument('--only_test', type=int, default=1, help="Split entire dataset or only test set")


def split_dataset(files, out_root):
    for i in tqdm(files):
        lf = []
        lf = glob(i+"*.json")
        all_pts = []

        for j in lf:
            meta = json.load(open(j))
            all_pts.append([meta['dot_xcam'], meta['dot_y_cam']])

        all_pts = np.array(all_pts)
        all_pts = np.round(all_pts, 3)
        un_pts = np.unique(all_pts, axis=0)
        cl_pts = np.zeros(len(all_pts))


        try:
            train_dots, test_dots = train_test_split(un_pts, test_size=0.3)
        except:
            # Not enough data
            continue

        for td in train_dots:
            cl_pts[np.where((td==all_pts).all(axis=1))[0]] = 1

        for td in test_dots:
            cl_pts[np.where((td==all_pts).all(axis=1))[0]] = 2

        for k in range(len(cl_pts)):
            of = out_root
            if(cl_pts[k] == 1):
                shutil.copy(lf[k], of+"/train/meta/")
                shutil.copy(lf[k].replace("meta", 'images').replace('json', 'jpg'), of+"/train/images/")
            elif(cl_pts[k] == 2):
                shutil.copy(lf[k], of+"/test/meta/")
                shutil.copy(lf[k].replace("meta", 'images').replace('json', 'jpg'), of+"/test/images/")
                
def add_ttv(path):
    os.mkdir(path+"/train/")
    os.mkdir(path+"/test/")
    
    os.mkdir(path+"/train/images/")
    os.mkdir(path+"/train/meta/")
    
    os.mkdir(path+"/test/images")
    os.mkdir(path+"/test/meta")
    
def prep_path(path, clear=True):
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
    add_ttv(path+"/")
    return path

def assign_work(path, out_dir, threads):
    procs = []
    all_files = glob(path)
    all_files = [i[:-11] for i in all_files]
    files = np.unique(all_files)
    
    print('Found ', len(all_files), ' meta from ', len(files), ' subjects.')
    
    chunk = len(files)//threads
    print(len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=split_dataset, args=(f, out_dir))
        procs.append(proc)
        proc.start()
        print("Start process #"+str(i))
        
    for proc in procs:
        proc.join()
        
def main():
    args = parser.parse_args()   
    prep_path(args.out_root)
    if(args.only_test):
        data_root = args.data_root+"/meta/*.json"
    else:
        data_root = args.data_root+"/*/meta/*.json"
        
    assign_work(data_root, args.out_root, args.threads)
    print("DONE")
    return 0   

if __name__ == "__main__":
    main()

