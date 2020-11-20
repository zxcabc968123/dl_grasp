import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pandas as pd
import random as rand
import datetime 
import argparse 

category = ''
photo_path = '/home/allen/dl_grasp/src/data_expend/origin_img/img/'+category+'/'
def main():
    print(photo_path)
    dirs = os.listdir(photo_path)
    img_num = len(dirs)

    for i in range(img_num):
        path_tmp = photo_path+category+'_'+str(i+1)+'.jpg'
        #path_check = photo_path+'check_'+category+'_'+str(i+1)+'.jpg'
        img_tmp = cv2.imread(path_tmp,3)
        #cv2.imshow('result_origin_img',img_tmp)
        h,w = img_tmp.shape[:2]
        for j in range(h):
            for k in range(w):
                img_tmp[j][k]=img_tmp[j][k]+22
        #cv2.imshow('result_gray_img',img_tmp)
        bgr_depth = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('result_gbgr_img',bgr_depth)
        # cv2.waitKey(0)
        cv2.imwrite(path_tmp,img_tmp)
        print('complete :',i+1)        

if __name__ == "__main__":
    main()