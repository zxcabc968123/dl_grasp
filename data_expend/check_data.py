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

parser = argparse.ArgumentParser()
parser.add_argument('--TrainDIR', type=str, default='', help='path to Trainingdata')
FLAGS = parser.parse_args()
data_csv='/home/allen/dl_grasp/src/data_expend'+FLAGS.TrainDIR
temp_path = '/home/allen/dl_grasp/src/data_expend/temp/'

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

def trans_degree(x,y,degree):
    degree=float(degree)*math.pi/180
    l=50
    f_x=float(x)+l*math.cos(degree)
    f_y=float(y)-l*math.sin(degree)
    tmp_x=x-round(f_x)
    tmp_y=y-round(f_y)
    return (round(tmp_x),round(tmp_y))
    
def main():
    data1,data2,data3,data4,data5,data6,data7,data8=pd_read_csv(data_csv)
    print('total data num :',len(data1))
    for i in range(len(data1)):
        print('Img :'+data1[i])
        print('target_x : {} terget_y : {}'.format(data2[i],data3[i]))
        print('angle :{}'.format(data4[i]))
        image=cv2.imread(data1[i],3)
        (a,b) = image.shape[:2]
        # for k in range(a):
        #     for j in range(b):
        #         if image[k][j][0]<10:
        #             image[k][j] = (243,243,243)
        cv2.circle(image, (data2[i],data3[i]), 5, (0,0,255),3)
        cv2.rectangle(image,(data5[i],data6[i]),(data7[i],data8[i])\
            ,(255, 0, 0),2)
        tmp_x,tmp_y=trans_degree(data2[i],data3[i],data4[i])
        cv2.line(image,(data2[i]+tmp_x,data3[i]+tmp_y),(data2[i]-tmp_x,data3[i]-tmp_y),(255,0,0),2)
        cv2.imshow('My image',image)
        #####
        save_path = temp_path + 'pit_' + str(i+1)+'.jpg'
        cv2.imwrite(save_path,image)
        #####
        cv2.waitKey(0)
    # img=cv2.imread('/home/allen/dl_grasp/src/data_expend/expand_img/data_12.jpg',0)
    # print(img.shape)
    # img=img.reshape((480,640,1))
    # print(img.shape)
    # print(img[227][361])
    # cv2.imshow('asfds',img)
    # cv2.waitKey(0)
    
if __name__ == "__main__":
    main()