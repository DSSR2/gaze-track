import cv2
import os
import json
import dlib
from glob import glob 
from multiprocessing import Pool, Process
from tqdm import tqdm

'''
Add eye landmarks to the MIT Gaze Capture Dataset using DLib
'''

import argparse
parser = argparse.ArgumentParser(description='Adding eye key points to meta data')
parser.add_argument('--dir', default='./dataset/', help='Path to converted dataset. Use dataset_converter')
parser.add_argument('--workers', default=40, type=int, help='Number of CPU cores available')
parser.add_argument('--p', default="./shape_predictor_68_face_landmarks.dat", help='Path to trained facial landmark model file')

def in_box(box, point):
    x1, y1, w, h = box
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def add_kps(files, p):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    no_face = 0
    err_ctr = 0
    buffer = 10
    for i in tqdm(files):
        img = cv2.imread(i)
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        meta = json.load(open(i.replace('images', 'meta').replace('.jpg', '.json')))
        screen_w, screen_h = meta['screen_w'], meta['screen_h']
        leye_x, leye_y, leye_w, leye_h = meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h']
        reye_x, reye_y, reye_w, reye_h = meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h']
        if(meta['face_valid']):
            fx, fy, fw, fh = meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h']
        else:
            no_face += 1
            faces = detector(bw_img, 1)
            for face in faces:
                if(face.left()<reye_x and face.top()<reye_y and face.right()>leye_x and face.bottom()>leye_y+leye_h):
                    fx, fy, fw, fh = face.left(), face.top(), face.right()-face.left, face.bottom()-face.top()
                    break
        
        face_rect = dlib.rectangle(fx, fy, fx+fw, fy+fh)
        kps = predictor(bw_img, face_rect)
        
        # make sure the detected landmark point are within the eye bounding box.
        if(in_box((reye_x-buffer, reye_y-buffer, reye_w+(buffer*2), reye_h+(buffer*2)), (kps.part(36).x, kps.part(36).y)) 
           and in_box((reye_x-buffer, reye_y-buffer, reye_w+(buffer*2), reye_h*(buffer*2)), (kps.part(39).x, kps.part(39).y))):
            
            meta['reye_x1'], meta['reye_y1'], meta['reye_x2'], meta['reye_y2'] = kps.part(36).x, kps.part(36).y, kps.part(39).x, kps.part(39).y
        else:
            err_ctr+=1
            meta['reye_x1'], meta['reye_y1'], meta['reye_x2'], meta['reye_y2'] = reye_x, reye_y+(reye_h//2), reye_x+reye_w, reye_y+(reye_h//2)
            
        if(in_box((leye_x-buffer, leye_y-buffer, leye_w+(buffer*2), leye_h+(buffer*2)), (kps.part(42).x, kps.part(42).y)) 
           and in_box((leye_x-buffer, leye_y-buffer, leye_w+(buffer*2), leye_h+(buffer*2)), (kps.part(45).x, kps.part(45).y))):
            meta['leye_x1'], meta['leye_y1'], meta['leye_x2'], meta['leye_y2'] = kps.part(42).x, kps.part(42).y, kps.part(45).x, kps.part(45).y
        else:
            err_ctr+=1
            meta['leye_x1'], meta['leye_y1'], meta['leye_x2'], meta['leye_y2'] = leye_x, leye_y+(leye_h//2), leye_x+leye_w, leye_y+(leye_h//2)
    
        meta_file = i.replace('images', 'meta').replace('.jpg', '.json')
        with open(meta_file, 'w') as outfile:
            json.dump(meta, outfile)
    print("face errs: ", no_face)
    
def assign_work(path, workers, p):
    procs = []
    files = glob(path+"/*/images/*.jpg")
    chunk = len(files)//workers
    print(len(files))
    
    for i in range(workers): 
        if(i==workers-1):
            f = files[i*chunk:]
        else:
            f = files[i*chunk:(i+1)*chunk]
        
        proc = Process(target=add_kps, args=(f, p))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()

def main():
    args = parser.parse_args()
    assign_work(args.dir, args.workers, args.p)
    print("KPs Added")
    
if __name__ == '__main__':
    main()
