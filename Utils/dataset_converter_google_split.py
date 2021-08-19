import json
import shutil
import numpy as np
from glob import glob
from multiprocessing import Pool, Process

'''
Convert MIT Gaze Capture dataset to the one used in "Accelerating eye movement research via accurate and affordable smartphone eye tracking" and prepare data for easy use in PyTorch.
1. Keep only portrait orientation images
2. Keep only iPhone images
3. Keep only images that have valid eye detections
4. Split data of each participant into train, test, split. 
'''

import argparse
parser = argparse.ArgumentParser(description='Convert the MIT Gaze Cap Dataset')
parser.add_argument('--dir', default="../../dataset/", help='Path to unzipped MIT dataset')
parser.add_argument('--out_dir', default="../../dataset/", help='Path to new dataset should have image, meta folders with train, val, test subfolders')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

def convert_dataset(files, out_root):
    for i in tqdm(files): 
        with open(i+"/info.json") as f:
            data = json.load(f)
            ds = data['Dataset']
            device = data['DeviceName']
        if(not('iPhone' in device)):
            continue
        
        expt_name = i.split('/')[-2]
        screen_info = json.load(open(i+'/screen.json'))
        face_det = json.load(open(i+'/appleFace.json'))
        l_eye_det = json.load(open(i+'/appleLeftEye.json'))
        r_eye_det = json.load(open(i+'/appleRightEye.json'))
        dot = json.load(open(i+'/dotInfo.json'))
        
        portrait_orientation = np.asarray(screen_info["Orientation"])==1
        l_eye_valid, r_eye_valid = np.array(l_eye_det['IsValid']), np.array(r_eye_det['IsValid'])
        valid_ids = portrait_orientation*l_eye_valid*r_eye_valid
        
        all_dots = np.asarray(dot['DotNum'])
        valid_dots = np.unique(all_dots[np.where(valid_ids==1)])
        
        try:
            train_dots, test_dots = train_test_split(valid_dots, test_size=0.2)
        except:
            # Too few dots to split, skip participant
            continue
            
            
        try:
            test_dots, val_dots = train_test_split(test_dots, test_size=0.4)
        except:
            val_dots = []
        
        split_ids = np.zeros(len(valid_ids))

        for j in train_dots:
            split_ids[all_dots==j] = 1

        for j in test_dots:
            split_ids[all_dots==j] = 2

        for j in val_dots:
            split_ids[all_dots==j] = 3
            
        split_ids = split_ids*valid_ids
        
        frame_ids = np.where(split_ids>0)[0]
        
        ctr = 0
        for frame_idx in frame_ids:
            fname = str(frame_idx).zfill(5)
            if(split_ids[frame_idx]==1):
                outdir = out_root+'/train/'
            elif(split_ids[frame_idx]==2):
                outdir = out_root+'/test/'
            elif(split_ids[frame_idx]==3):
                outdir = out_root+'/val/'
            
            meta = {}
            meta['device'] = device
            meta['screen_h'], meta['screen_w'] = screen_info["H"][frame_idx], screen_info["W"][frame_idx]
            meta['face_valid'] = face_det["IsValid"][frame_idx]
            meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h'] = round(face_det['X'][frame_idx]), round(face_det['Y'][frame_idx]), round(face_det['W'][frame_idx]), round(face_det['H'][frame_idx])
            meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h'] = meta['face_x']+round(l_eye_det['X'][frame_idx]), meta['face_y']+round(l_eye_det['Y'][frame_idx]), round(l_eye_det['W'][frame_idx]), round(l_eye_det['H'][frame_idx])
            meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h'] = meta['face_x']+round(r_eye_det['X'][frame_idx]), meta['face_y']+round(r_eye_det['Y'][frame_idx]), round(r_eye_det['W'][frame_idx]), round(r_eye_det['H'][frame_idx])
            meta['dot_xcam'], meta['dot_y_cam'] = dot['XCam'][frame_idx], dot['YCam'][frame_idx]
            meta['dot_x_pix'], meta['dot_y_pix'] = dot['XPts'][frame_idx], dot['YPts'][frame_idx]

            shutil.copy(i+'/frames/'+fname+".jpg", outdir+"/images/"+expt_name+'__'+fname+'.jpg')            
            meta_file = outdir+'/meta/'+expt_name+'__'+fname+'.json'
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)
            ctr+=1
    return 0

def assign_work(path, out_dir, threads):
    procs = []
    files = glob(path+"/*/")
    chunk = len(files)//threads
    print(len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=convert_dataset, args=(f, out_dir))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

def main():
    args = parser.parse_args()
    assign_work(args.dir, args.out_dir, args.threads)
    print("Conversion Complete")

if __name__=="__main__":
    main()
