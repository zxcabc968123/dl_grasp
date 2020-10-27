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
import random as rand

origin_data_csv = '/home/allen/dl_grasp/src/data_expend/origin_data/blackbox_2020-10-23_15_08_17_.csv'
expand_folder = '/home/allen/dl_grasp/src/data_expend/expand_data'


def rotate_image(roi,origin_degree):
    degree=rand.randint(0,360)
    size=rand.uniform(0.5,1.5)
    print('degree : {}'.format(degree))
    (h, w) = roi.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degree,size )

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    roi =cv2.warpAffine(roi, M, (nW, nH), flags=cv2.INTER_NEAREST,borderValue=3)
    #cv2.imshow('rotate_img_1',roi)
    (h, w) = roi.shape[:2]
    for i in range(h):
        for j in range(w):
            if roi[i][j]<=10:
                roi[i][j]=rand.randint(243,245)
    cv2.imshow('rotate_img_2',roi)
    #####Dilation 
    kernel = np.ones((3,3), np.uint8)
    roi = cv2.erode(roi, kernel, iterations = 3)
    cv2.imshow('rotate_img_3',roi)
    cv2.waitKey(0)

    return roi

def load_background():
    background_folder = '/home/allen/dl_grasp/src/data_expend/background/'
    dirs = os.listdir(background_folder)
    background_img=background_folder+'background_'+str(rand.randint(1,len(dirs)))+'.jpg'
    return background_img
def pd_read_csv(csvFile):
 
    data_frame = pd.read_csv(csvFile, sep=",")
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    data8 = []

    for index in data_frame.index:
        data_row = data_frame.loc[index]
        data1.append(data_row["image_path"])
        data2.append(data_row["target_x"])
        data3.append(data_row["target_y"])
        data4.append(data_row["target_angle"])
        data5.append(data_row["ix"])
        data6.append(data_row["iy"])
        data7.append(data_row["rx"])
        data8.append(data_row["ry"])
    return (data1,data2,data3,data4,data5,data6,data7,data8)
def main():
    image_path,target_x,target_y,target_angle,ix,iy,rx,ry=pd_read_csv(origin_data_csv)
    image=cv2.imread(image_path[0],0)
    cv2.imshow('origin_image',image)
    cut_image=image[iy[0]:ry[0],ix[0]:rx[0]]
    h_img=ry[0]-iy[0]
    w_img=rx[0]-ix[0]
    print('h_img:{} w_img:{}'.format(h_img,w_img))
    #cv2.imshow('cut_image',cut_image)
    rotate_img=rotate_image(cut_image,target_angle[0])
    (h_img, w_img) = rotate_img.shape[:2]
    cv2.imshow('rotate_image',rotate_img)
    background_img_path=load_background()
    print(background_img_path) 
    background_img=cv2.imread(background_img_path,0)
    #cv2.imshow('background_image',background_img)
    new_ix=rand.randint(0,640-w_img)
    new_iy=rand.randint(0,480-h_img)
    background_img[new_iy:new_iy+h_img,new_ix:new_ix+w_img]=rotate_img
    cv2.imshow('new_img',background_img)
    print(background_img.shape)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()