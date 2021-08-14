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
Split the test set of the converted dataset into SVR trianing and testing. 
Input should be root of folder containing dataset converted using dataset_converter_*.py
'''


