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
Split the test set of the converted dataset into 13 point SVR training and testing. 
Input should be root of folder containing dataset converted using dataset_converter_*.py and root directory of completly unzipped GazeCapture dataset. 
'''

import argparse
parser = argparse.ArgumentParser(description='Split Dataset based on device')
parser.add_argument('--mit_root', help="Path to unzipped MIT GazeCapture dataset. Should have 5 digit folder names")
parser.add_argument('--data_root', help="Path to a dataset converted using any of the dataset_converter_*.py files")
parser.add_argument('--out_root', help="Path to store the new split test set")
parser.add_argument('--only_test', default=1, type=int, help="Split whole dataset or only test set")
parser.add_argument('--threads', type=int, help="Number of workers")

def split_dataset(root, folders, data_root, new_root):
    pts = []
    for i in tqdm(folders):
        part = i[-6:-1]
        data = json.load(open(root+"/"+part+"/dotInfo.json"))

        screen_info = json.load(open(root+"/"+part+"/screen.json"))

        port = np.asarray(screen_info["Orientation"])==1

        pts = [(data['XCam'][i], data['YCam'][i]) for i in range(len(data['XCam']))]
        pts = np.array(pts)[np.where(port==1)]
        pts = np.around(pts, 7)
        un_pts = np.unique(pts, axis=0)

        calib_pts_nine = np.array([[un_pts.min(axis=0)[0], un_pts.min(axis=0)[1]],
                                   [un_pts.min(axis=0)[0], un_pts.max(axis=0)[1]],
                                   [un_pts.max(axis=0)[0], un_pts.min(axis=0)[1]],
                                   [un_pts.max(axis=0)[0], un_pts.max(axis=0)[1]],

                                   [(un_pts.max(axis=0)[0]+un_pts.min(axis=0)[0])/2, (un_pts.max(axis=0)[1]+un_pts.min(axis=0)[1])/2],

                                   [un_pts.max(axis=0)[0], (un_pts.max(axis=0)[1]+un_pts.min(axis=0)[1])/2],
                                   [un_pts.min(axis=0)[0], (un_pts.max(axis=0)[1]+un_pts.min(axis=0)[1])/2],
                                   [(un_pts.max(axis=0)[0]+un_pts.min(axis=0)[0])/2, un_pts.max(axis=0)[1]],
                                   [(un_pts.max(axis=0)[0]+un_pts.min(axis=0)[0])/2, un_pts.min(axis=0)[1]]
                                  ])

        calib_pts_thirteen = np.array([[(calib_pts_nine[0][0]+calib_pts_nine[8][0])/2, (calib_pts_nine[0][1]+calib_pts_nine[6][1])/2],
                                       [(calib_pts_nine[0][0]+calib_pts_nine[8][0])/2, (calib_pts_nine[1][1]+calib_pts_nine[6][1])/2],
                                       [(calib_pts_nine[2][0]+calib_pts_nine[8][0])/2, (calib_pts_nine[0][1]+calib_pts_nine[6][1])/2],
                                       [(calib_pts_nine[2][0]+calib_pts_nine[8][0])/2, (calib_pts_nine[1][1]+calib_pts_nine[6][1])/2]
                                      ])
        calib_pts_thirteen = np.append(calib_pts_nine, calib_pts_thirteen, axis=0)
        calib_pts_thirteen = np.around(calib_pts_thirteen, 7)

        fids = np.zeros(len(pts))
        pt_num = np.zeros(len(pts))
        pt_ctr = 1

        for b in range(13):
            fids[np.where((calib_pts_thirteen[b]==pts).all(axis=1))[0]] = 1
            pt_num[np.where((calib_pts_thirteen[b]==pts).all(axis=1))[0]] = pt_ctr
            pt_ctr += 1

        for frame_idx in range(len(fids)):
            fname = part+"__"+str(frame_idx).zfill(5)
            try:
                meta = json.load(open(data_root+"/meta/"+fname+".json"))
                meta['dotNum'] = int(pt_num[frame_idx])
                with open(data_root+"/meta/"+fname+".json", 'w') as outfile:
                    json.dump(meta, outfile)

                if(fids[frame_idx] == 0):
                    shutil.copy(data_root+"/images/"+fname+".jpg", new_root+"/test/images/")
                    shutil.copy(data_root+"/meta/"+fname+".json", new_root+"/test/meta/")

                elif(fids[frame_idx] == 1):
                    shutil.copy(data_root+"/images/"+fname+".jpg", new_root+"/train/images/")
                    shutil.copy(data_root+"/meta/"+fname+".json", new_root+"/train/meta/")

            except:
                pass

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

def main():
    args = parser.parse_args()
    threads = args.threads
    prep_path(args.out_root)
    
    procs = []
    if(args.only_test):
        files = glob(args.data_root+"/meta/*.json")
    else:
        files = glob(args.data_root+"/*/meta/*.json")
        
    chunk = len(files)//threads
    folders = glob(args.mit_root+"/*/")
    
    print("Number of files: ", len(files))
    for i in range(threads): 
        f = folders[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = folders[i*chunk:]
        
        proc = Process(target=split_dataset, args=(args.mit_root, f, args.data_root, args.out_root))
        procs.append(proc)
        proc.start()
        print("Start process #"+str(i))
        
    for proc in procs:
        proc.join()
        
    print("DONE")
    return 0

if __name__ == '__main__':
    main()