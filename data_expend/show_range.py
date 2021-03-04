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

# parser = argparse.ArgumentParser()
# parser.add_argument('--TrainDIR', type=str, default='', help='path to Trainingdata')
# FLAGS = parser.parse_args()
data_csv='/home/allen/dl_grasp/src/data_expend/expand_data/data_2020-11-20_05_04_21_.csv'
data_csv='/home/allen/dl_grasp/src/data_expend/origin_data/1119_14object/14_object_num20_1119.csv'
image_path = '/home/allen/dl_grasp/src/data_expend/notrain39_rgb.jpg'
save_path = '/home/allen/dl_grasp/src/data_expend/range_plot/'

def trans_degree(x,y,degree):
    print(degree)
    degree=float(degree)*math.pi/180
    l=100
    #####
    f_x=float(x)-l*math.sin(degree)
    f_y=float(y)+l*math.cos(degree)
    #####
    tmp_x=x-round(f_x)
    tmp_y=y-round(f_y)
    return (round(tmp_x),round(tmp_y))

def main():
    # data1,data2,data3,data4,data5,data6,data7,data8=pd_read_csv(data_csv)
    # for i in range(len(data1)):
    #     print('Img :'+data1[i])
    #     print('target_x : {} terget_y : {}'.format(data2[i],data3[i]))
    #     print('angle :{}'.format(data4[i]))
    #     image=cv2.imread(data1[i],3)
    #     print(image.shape)
    #     cv2.circle(image, (int(data2[i]),int(data3[i])), 5, (0,0,255),3)
    #     cv2.rectangle(image,(data5[i],data6[i]),(data7[i],data8[i])\
    #          ,(255, 0, 0),2)
    #     tmp_x,tmp_y=trans_degree(data2[i],data3[i],data4[i])
    #     cv2.line(image,(int(data2[i]+tmp_x),int(data3[i]+tmp_y)),(int(data2[i]-tmp_x),int(data3[i]-tmp_y)),(255,0,0),2)
    #     cv2.imshow('My image',image)

    #     cv2.waitKey(0)
    rect_range = 10
    img_1=cv2.imread(image_path,3)
    cv2.circle(img_1, (int(342),int(265)), 3, (0,0,255),4)
    cv2.rectangle(img_1,(342+ rect_range,265+rect_range),(342-rect_range,265-rect_range),(0, 255, 0),2)
    cv2.imshow('img_1',img_1)
    cv2.imwrite(save_path+str(rect_range)+'.jpg',img_1)

    rect_range = 25
    img_2=cv2.imread(image_path,3)
    cv2.circle(img_2, (int(342),int(265)), 3, (0,0,255),4)
    cv2.rectangle(img_2,(342+ rect_range,265+rect_range),(342-rect_range,265-rect_range),(0, 255, 0),2)
    cv2.imshow('img_2',img_2)
    cv2.imwrite(save_path+str(rect_range)+'.jpg',img_2)

    rect_range = 40
    img_3=cv2.imread(image_path,3)
    cv2.circle(img_3, (int(342),int(265)), 3, (0,0,255),4)
    cv2.rectangle(img_3,(342+ rect_range,265+rect_range),(342-rect_range,265-rect_range),(0, 255, 0),2)
    cv2.imshow('img_3',img_3)
    cv2.imwrite(save_path+str(rect_range)+'.jpg',img_3)

    ang_range = 75
    leng = 50
    img_4=cv2.imread(image_path,3)
    #cv2.circle(img_4, (int(342),int(265)), 3, (0,0,255),4)
    
    #####
    temp_x,temp_y=trans_degree(342,265,ang_range)
    # cv2.line(img_4,(int(342+temp_x),int(265+temp_y)),(int(342),int(265)),(0,255,0),3)
    # cv2.line(img_4,(int(342+temp_x),int(265-temp_y)),(int(342),int(265)),(0,255,0),3)

    # cv2.line(img_4,(int(342-temp_x),int(265+-temp_y)),(int(342),int(265)),(0,255,255),3)
    # cv2.line(img_4,(int(342-temp_x),int(265+temp_y)),(int(342),int(265)),(0,255,255),3)
    #####
    cv2.line(img_4,(int(342+temp_x),int(265+temp_y)),(int(342-temp_x),int(265-temp_y)),(0,255,0),3)
    cv2.line(img_4,(int(342-temp_x),int(265+temp_y)),(int(342+temp_x),int(265-temp_y)),(0,255,255),3)
    #####
    cv2.line(img_4,(int(342+leng),int(265)),(int(342-leng),int(265)),(0,0,255),3)
    cv2.imshow('img_4',img_4)
    cv2.imwrite(save_path+'line_1'+'.jpg',img_4)

    ###
    img_5=cv2.imread(image_path,3)
    cv2.line(img_5,(int(342+temp_x),int(265+temp_y)),(int(342),int(265)),(0,255,0),3)
    cv2.line(img_5,(int(342+temp_x),int(265-temp_y)),(int(342),int(265)),(0,255,0),3)

    cv2.line(img_5,(int(342-temp_x),int(265+-temp_y)),(int(342),int(265)),(0,255,255),3)
    cv2.line(img_5,(int(342-temp_x),int(265+temp_y)),(int(342),int(265)),(0,255,255),3)

    cv2.line(img_5,(int(342+leng),int(265)),(int(342-leng),int(265)),(0,0,255),3)

    cv2.imshow('img_5',img_5)
    cv2.imwrite(save_path+'line_2'+'.jpg',img_5)

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
