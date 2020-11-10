#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
import math
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

net_path = '/home/allen/dl_grasp/src/train/Save_net/CNN_MSE201030'
data_csv = '/home/allen/dl_grasp/src/data_expend/expand_data/40data_2020-10-29_16_16_22_.csv'

angle_range = 30
x_locate_range = 20
y_locate_range = 20


def create_result_array(data2,data3,data4):

    result_array=np.zeros([len(data2),3],  dtype=float)
    for i in range(len(data2)):
        result_array[i]=[data2[i],data3[i],data4[i]]
    return result_array

def create_photo_array(data1):

    photo_array= cv2.imread(data1[0],0)
    for i in range(len(data1)-1):
        image=cv2.imread(data1[i+1],0)
        photo_array=np.concatenate((photo_array,image))
    photo_array=photo_array.reshape((-1,480,640,1))
    ########normalize
    photo_array=photo_array/255
    return photo_array

def trans_degree(x,y,degree):
    degree=float(degree)*math.pi/180
    l=50
    f_x=float(x)+l*math.cos(degree)
    f_y=float(y)-l*math.sin(degree)
    tmp_x=x-round(f_x)
    tmp_y=y-round(f_y)
    return (round(tmp_x),round(tmp_y))

def pd_read_csv(csvFile):
 
    data_frame = pd.read_csv(csvFile, sep=",")
    data1 = []
    data2 = []
    data3 = []
    data4 = []
   
    for index in data_frame.index:
        data_row = data_frame.loc[index]
        data1.append(data_row["image_path"])
        data2.append(data_row["target_x"])
        data3.append(data_row["target_y"])
        data4.append(data_row["target_angle"])

    return (data1,data2,data3,data4)

def get_test_image(image_path,reload_sm_keras):
    image=cv2.imread(image_path,0)  
    image_data=image.reshape(-1,480,640,1)
    image_data=image_data/255
    result_point=reload_sm_keras.predict([image_data])
    cv2.circle(image, (result_point[0][0],result_point[0][1]),5,  (0, 0, 255),2)
    temp_x,temp_y=trans_degree(result_point[0][0],result_point[0][1],result_point[0][2])
    cv2.line(image,(int(result_point[0][0]+temp_x),int(result_point[0][1]+temp_y)),(int(result_point[0][0]-temp_x),int(result_point[0][1]-temp_y)),(0,0,255),2)
    ###
    #cv2.imshow('My Image', image)
    print('prdict locate : x: {}  y: {} z: {}'.format(result_point[0][0],result_point[0][1],result_point[0][2]))
    #cv2.waitKey(0)
    return image

def main():

    reload_sm_keras = tf.keras.models.load_model(net_path)
    #score = model.evaluate(x_test, y_test, verbose=0)
    data1,data2,data3,data4=pd_read_csv(data_csv)
    validation_input=create_photo_array(data1)
    validation_output=create_result_array(data2,data3,data4)
    score = reload_sm_keras.evaluate(validation_input, validation_output,batch_size=1,verbose=1)
    print("test loss, test acc:", score)
    #####print 40 voalidation result
    difference = [0.0, 0.0, 0.0]
    accurate_num = 0.0
    pre_time = 0.0
    for i in range(len(validation_input)):
        temp_input=validation_input[i].reshape(-1,480,640,1)
        ##########predict time#######
        t0 = time.time()
        predict_point=reload_sm_keras.predict([temp_input])
        pre_time += (time.time() - t0)
        #############################
        difference[0] += abs(data2[i]-predict_point[0][0])
        difference[1] += abs(data3[i]-predict_point[0][1])
        difference[2] += abs(data4[i]-predict_point[0][2])
        if (abs(data2[i]-predict_point[0][0])<=x_locate_range )&(abs(data3[i]-predict_point[0][1])<=y_locate_range)&(abs(data3[i]-predict_point[0][2])<=angle_range):
            accurate_num+=1
    #print('accurate_rate : {}'.format(accurate_num))
    print('accurate_num : {}'.format(accurate_num/len(validation_input)*100))
    print('x_mae : {}'.format(difference[0]/len(validation_input)))
    print('y_mae : {}'.format(difference[1]/len(validation_input)))
    print('angle_mae : {}'.format(difference[2]/len(validation_input)))
    print('all_mae : {}'.format((difference[0]+difference[1]+difference[2])/len(validation_input)))
    print('average_time : {}'.format(pre_time/len(validation_input)))

    for i in range(len(validation_input)):
        print('ground true locate : x: {}  y: {} z: {}'.format(data2[i],data3[i],data4[i]))
        result_img=get_test_image(data1[i],reload_sm_keras)
        
        ######
        cv2.circle(result_img, (int(data2[i]),int(data3[i])),5,  (255, 0, 0),2)
        temp_x,temp_y=trans_degree(data2[i],data3[i],data4[i])
        cv2.line(result_img,(int(data2[i]+temp_x),int(data3[i]+temp_y)),(int(data2[i]-temp_x),int(data3[i]-temp_y)),(255,0,0),2)
        ######
        cv2.imshow('result_img',result_img)
        cv2.waitKey(0)
if __name__ == "__main__":
    main()
