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
            train_dots, test_dots = train_test_split(un_pts, test_size=0.25)
        except:
            train_dots = un_pts
            test_dots = []
        try:
            test_dots, val_dots = train_test_split(test_dots, test_size=0.3)
        except:
            val_dots = []

        for td in train_dots:
            cl_pts[np.where((td==all_pts).all(axis=1))[0]] = 1

        for td in test_dots:
            cl_pts[np.where((td==all_pts).all(axis=1))[0]] = 2

        for td in val_dots:
            cl_pts[np.where((td==all_pts).all(axis=1))[0]] = 3

        for k in range(len(cl_pts)):
            of = out_root
            if(cl_pts[k] == 1):
                shutil.copy(lf[k], of+"/train/meta/")
                shutil.copy(lf[k].replace("meta", 'images').replace('json', 'jpg'), of+"/train/images/")
            elif(cl_pts[k] == 2):
                shutil.copy(lf[k], of+"/test/meta/")
                shutil.copy(lf[k].replace("meta", 'images').replace('json', 'jpg'), of+"/test/images/")
            elif(cl_pts[k] == 3):
                shutil.copy(lf[k], of+"/val/meta/")
                shutil.copy(lf[k].replace("meta", 'images').replace('json', 'jpg'), of+"/val/images/")        


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
        
        proc = Process(target=move_files, args=(f, out_dir))
        procs.append(proc)
        proc.start()
        print(i)
        
    for proc in procs:
        proc.join()
        
def main():
    args = parser.parse_args()    
    if(args.only_test):
        data_root = args.data_root+"/meta/*.json"
    else:
        data_root = args.data_root+"/*/meta/*.json"
        
    assign_work(data_root, args.out_root, args.threads)
    print("DONE")
    return 0   

if __name__ == "__main__":
    main()

