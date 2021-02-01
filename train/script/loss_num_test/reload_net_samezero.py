#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras.backend as kb

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

#net_path = '/home/allen/dl_grasp/src/train/Save_net/CNN_MSE201105_adjustdegree_normalize_nodrop_losschange_dropout'
#net_path = '/home/allen/dl_grasp/src/train/Save_net/14object/drop/1120_14object_dropoutv5'
net_path = '/home/allen/dl_grasp/src/train/Save_net/5object_lossnum_check/mse_num12'
#data_csv = '/home/allen/dl_grasp/src/data_expend/expand_data/40data_2020-10-29_16_16_22_.csv'
data_csv = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_200_0126.csv'
#data_csv = '/home/allen/dl_grasp/src/data_expend/expand_data/1000blackdata_2020-10-28_07_13_23_.csv'

angle_range = 15
x_locate_range = 25
y_locate_range = 25

def custom_loss(y_actual,y_pred):
    x_gap = tf.square(y_pred[:,0]-y_actual[:,0])
    y_gap = tf.square(y_pred[:,1]-y_actual[:,1])
    cos_gap = tf.square(y_pred[:,2]-y_actual[:,2])
    sin_gap =  tf.square(y_pred[:,3]-y_actual[:,3])

    loss = 1.2*x_gap + 1.2*y_gap + cos_gap + sin_gap

    #return tf.math.sqrt(tf.math.reduce_mean(loss))
    return tf.math.reduce_sum(loss)
    
def arctan_recovery(cos_x,sin_x):
    #print('cos_x : {},sin_x : {} '.format(cos_x,sin_x))
    predict_degree = 0.5*np.arctan2(sin_x,cos_x)
    predict_degree=predict_degree/math.pi*180
    # if (predict_degree <0):
    #     predict_degree=predict_degree
    # #print('arctan_degree : {}'.format(predict_degree))
    return predict_degree

def create_result_array(data2,data3,data4):

    result_array=np.zeros([len(data2),4],  dtype=float)

    for i in range(len(data2)):
        diameter = data4[i]*math.pi/180
        cos_pi = math.cos(2*diameter)
        sin_pi = math.sin(2*diameter)
        result_array[i]=[data2[i]/320,data3[i]/240,cos_pi,sin_pi]
        #print('cos : {} sin : {}'.format(cos_pi,sin_pi))
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
    print(degree)
    degree=float(degree)*math.pi/180
    l=50
    #####
    f_x=float(x)-l*math.sin(degree)
    f_y=float(y)+l*math.cos(degree)
    #####
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
        ######################
    for i in range(len(data2)):
        data2[i] = data2[i]-320
        data3[i] = data3[i]-240
        ######################
    for i in range(len(data4)):
         #######################
        if data4[i]>90:
            data4[i] = -1*(data4[i]-90)
        elif data4[i]<=90:
            data4[i] = 90-data4[i]
    print(data4)
    return (data1,data2,data3,data4)

def get_test_image(image_path,reload_sm_keras):
    image=cv2.imread(image_path,0)  
    image_data=image.reshape(-1,480,640,1)
    image_data=image_data/255
    result_point=reload_sm_keras.predict([image_data])
    ########
    result_point[0][0]=result_point[0][0]*320+320
    result_point[0][1]=result_point[0][1]*240+240
    cv2.circle(image, (result_point[0][0],result_point[0][1]),5,  (0, 0, 255),2)
    degree = arctan_recovery(result_point[0][2],result_point[0][3])

    temp_x,temp_y=trans_degree(result_point[0][0]*640,result_point[0][1]*480,degree)
    cv2.line(image,(int(result_point[0][0]+temp_x),int(result_point[0][1]+temp_y)),(int(result_point[0][0]-temp_x),int(result_point[0][1]-temp_y)),(0,0,255),2)
    ###
    #cv2.imshow('My Image', image)
    print('prdict locate : x: {}  y: {} degree: {}'.format(result_point[0][0],result_point[0][1],degree))
    #cv2.waitKey(0)
    return image

def main():

    reload_sm_keras = tf.keras.models.load_model(net_path,custom_objects={'custom_loss': custom_loss})
    reload_sm_keras.summary()
    #score = model.evaluate(x_test, y_test, verbose=0)
    data1,data2,data3,data4=pd_read_csv(data_csv)
    validation_input=create_photo_array(data1)
    validation_output=create_result_array(data2,data3,data4)
    score = reload_sm_keras.evaluate(validation_input, validation_output,batch_size=1,verbose=1)
    print("test loss, test acc:", score)
    #####print 40 voalidation result
    difference = [0.0, 0.0, 0.0]
    pre_time = 0.0
    accurate_num = 0
    for i in range(len(validation_input)):
        temp_input=validation_input[i].reshape(-1,480,640,1)
        ##########predict time#######
        t0 = time.time()
        predict_point=reload_sm_keras.predict([temp_input])
        predict_point[0][0]=predict_point[0][0]*320
        predict_point[0][1]=predict_point[0][1]*240
        pre_time += (time.time() - t0)
        #############################
        difference[0] += abs(data2[i]-(predict_point[0][0]))
        difference[1] += abs(data3[i]-(predict_point[0][1]))
        ###########calculate arctan#########
        predict_degree=arctan_recovery(predict_point[0][2],predict_point[0][3])
        predict_degree_array = np.array([predict_degree , 0.0])
        if predict_degree>=0:
            predict_degree_array=np.array([predict_degree,predict_degree-180])
        elif predict_degree<0:
            predict_degree_array=np.array([predict_degree,predict_degree+180])
        #############################
        difference[2] += abs(data4[i]-predict_degree)

        if (abs(data2[i]-predict_point[0][0])<=x_locate_range )&(abs(data3[i]-predict_point[0][1])<=y_locate_range)&\
            ((abs(data4[i]-predict_degree_array[0])<=angle_range)or(abs(data4[i]-predict_degree_array[1])<=angle_range)):
            accurate_num=accurate_num+1
    print('limit : x: {} y: {} angle: {}'.format(x_locate_range,y_locate_range,angle_range))
    print('net_path : {}'.format(net_path))
    print('accurate_num : {}'.format(accurate_num/len(validation_input)*100))
    print('x_mae : {}'.format(difference[0]/len(validation_input)))
    print('y_mae : {}'.format(difference[1]/len(validation_input)))
    print('angle_mae : {}'.format(difference[2]/len(validation_input)))
    print('all_mae : {}'.format((difference[0]+difference[1]+difference[2])/len(validation_input)))
    print('average_time : {}'.format(pre_time/len(validation_input)))

    for i in range(len(validation_input)):
        print('ground true locate : x: {}  y: {} degree: {}'.format(data2[i],data3[i],data4[i]))
        print('ground true cos : {} sin : {} '.format(math.cos(2*data4[i]*math.pi/180),math.sin(2*data4[i]*math.pi/180)))
        result_img=get_test_image(data1[i],reload_sm_keras)
        ######
        cv2.circle(result_img, (int(data2[i]+320),int(data3[i]+240)),5,  (255, 0, 0),2)
        temp_x,temp_y=trans_degree(data2[i]+320,data3[i]+240,data4[i])
        cv2.line(result_img,(int(data2[i]+320+temp_x),int(data3[i]+240+temp_y)),(int(data2[i]+320-temp_x),int(data3[i]+240-temp_y)),(255,0,0),2)
        #cv2.line(result_img,(int(data2[i]+temp_x),int(data3[i]+temp_y)),(int(data2[i]),int(data3[i])),(255,0,0),2)
        ######
        cv2.imshow('result_img',result_img)
        cv2.waitKey(0)
if __name__ == "__main__":
    main()
