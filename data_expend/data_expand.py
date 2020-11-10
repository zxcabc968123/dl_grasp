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
import datetime 
import argparse 
day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]

current_time = day + "_" + time + "_"
####save
expand_data_path = "/home/allen/dl_grasp/src/data_expend/expand_data/"
expand_img_path = "/home/allen/dl_grasp/src/data_expend/expand_img/"

parser = argparse.ArgumentParser()

parser.add_argument('--expand_time', type=int,help='expand times of each data')
FLAGS = parser.parse_args()

expands_times = FLAGS.expand_time
#######################################################################################################
                                            #origin_data_csv
origin_data_csv = '/home/allen/dl_grasp/src/data_expend/origin_data/blackbox_2020-10-23_15_08_17_.csv'

########################################################################################################
def save_information(fin_img,fin_x,fin_y,fin_degree):
    dirs = os.listdir(expand_img_path)
    img_num = len(dirs)
    img_path = expand_img_path + 'data_'+str(img_num+1) + '.jpg'
    cv2.imwrite(img_path,fin_img)
    line =img_path + ',' + str(fin_x) + ',' + str(fin_y) + ','+ str(fin_degree)+ '\n'

    if not os.path.exists(expand_data_path):
        os.makedirs(expand_data_path)            
    with open(expand_data_path + 'data' + "_" + current_time + '.csv', 'a') as f:
        f.writelines(line)
        print('Save data : {}'.format(img_num+1))

###range: 0.9~1.1
def stretch_img(roi):
    size=rand.uniform(0.9,1)
    (h, w) = roi.shape[:2]
    for i in range(h):
        for j in range(w):
            if not (roi[i][j]>230):
                roi[i][j]=roi[i][j]*size
    return roi
###range: 0~360
def rotate_image(roi,origin_degree): 
    degree=rand.randint(0,360)
    ###adjust grasp degree###
    new_grasp_degree=origin_degree+degree
    if  new_grasp_degree>=360:
        new_grasp_degree=new_grasp_degree-360
    if  new_grasp_degree>=180:
        new_grasp_degree=new_grasp_degree-180
    size=rand.uniform(0.7,1.2)
    #print('rotating_degree : {}'.format(degree))
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
    #####Dilation 
    kernel = np.ones((3,3), np.uint8)
    roi = cv2.erode(roi, kernel, iterations = 3)

    return roi,new_grasp_degree

def load_background():
    background_folder = '/home/allen/dl_grasp/src/data_expend/background/'
    dirs = os.listdir(background_folder)
    background_img_path=background_folder+'background_'+str(rand.randint(1,len(dirs)))+'.jpg'
    background_img=cv2.imread(background_img_path,0)
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
def slide_img(roi):
    (h_img, w_img) = roi.shape[:2]
    new_ix=rand.randint(0,640-w_img)
    new_iy=rand.randint(0,480-h_img)
    background_img=load_background()
    background_img[new_iy:new_iy+h_img,new_ix:new_ix+w_img]=roi
    fin_x = new_ix+0.5*w_img
    fin_y = new_iy+0.5*h_img
    return background_img,fin_x,fin_y
    
def main():
    image_path,target_x,target_y,target_angle,ix,iy,rx,ry=pd_read_csv(origin_data_csv)
    for i in range (len(image_path)):
        image=cv2.imread(image_path[i],0)
        for j in range(expands_times):     
            cut_image=image[iy[i]:ry[i],ix[i]:rx[i]]
            h_img=ry[i]-iy[i]
            w_img=rx[i]-ix[i]
            (rotate_img,new_grasp_degree)=rotate_image(cut_image,target_angle[i])
            new_img=stretch_img(rotate_img)
            (fin_img,fin_x,fin_y)=slide_img(new_img)
            #print('new grasp x:{} y:{} degree : {}'.format(fin_x,fin_y,new_grasp_degree))
            ###save
            save_information(fin_img,fin_x,fin_y,new_grasp_degree)



if __name__ == "__main__":
    main()