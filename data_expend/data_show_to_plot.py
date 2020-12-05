#!/usr/bin/env python3
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
import argparse
import math
import random as rand
img_path = '/home/allen/dl_grasp/src/data_expend/origin_img/img/blackbox/blackbox_22.jpg'
#img_path = '/home/allen/dl_grasp/src/data_expend/origin_img/img/pen/pen_13.jpg'
img_path = '/home/allen/dl_grasp/src/data_expend/background/background_13.jpg'
store_path = '/home/allen/dl_grasp/src/data_expend/fordeal_temp'
def main():
    image=cv2.imread('/home/allen/dl_grasp/src/data_expend/origin_img/img/bottle/bottle_11.jpg',0)
    cv2.imwrite(store_path+'/bottle_origin.jpg',image)
    image=cv2.imread(img_path,0)
    cv2.imwrite(store_path+'/origin.jpg',image)
    cv2.imshow('My image',image)
    (h, w) = image.shape[:2]
    for i in range(h):
        for j in range(w):
            if image[i][j]<=80:
                image[i][j]=rand.randint(243,245)
    cv2.imshow('delete showdow',image)
    cv2.imwrite(store_path+'/showdow.jpg',image)
    ##########################
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations = 5)
    cv2.imshow('erode',image)
    cv2.imwrite(store_path+'/erode.jpg',image)
    ##########################
    kernel = np.ones((5,5), np.uint8)
    image = cv2.erode(image, kernel, iterations = 3)
    cv2.imshow('dilate',image)
    cv2.imwrite(store_path+'/dilate.jpg',image)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()