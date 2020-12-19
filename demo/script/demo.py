#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(2, "/home/allen/realsensepkg/catkin_workspace/install/lib/python3/dist-packages")
import rospy
import cv2
from get_rs_image import Get_image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pynput import keyboard
import time
import math
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
import random as rand
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import cv2
from cv_bridge import CvBridge, CvBridgeError

net_path = '/home/allen/dl_grasp/src/train/Save_net/14object/drop/1120_14object_dropoutv5'

def plot_result(img,ix,iy,tx,ty):
    cv2.circle(img, (int(ix),int(iy)),5,(0, 255, 0),5)
    cv2.line(img,(int(ix+tx),int(iy+ty)),(int(ix-tx),int(iy-ty)),(0,0,255),5)
    return img

def trans_degree(x,y,degree):
    #print(degree)
    degree=float(degree)*math.pi/180
    l=50
    #####
    f_x=float(x)-l*math.sin(degree)
    f_y=float(y)+l*math.cos(degree)
    #####
    tmp_x=x-round(f_x)
    tmp_y=y-round(f_y)
    return (round(tmp_x),round(tmp_y))

def arctan_recovery(cos_x,sin_x):
    predict_degree = 0.5*np.arctan2(sin_x,cos_x)
    predict_degree=predict_degree/math.pi*180
    return predict_degree


def custom_loss(y_actual,y_pred):
    x_gap = tf.square(y_pred[:,0]-y_actual[:,0])
    y_gap = tf.square(y_pred[:,1]-y_actual[:,1])
    cos_gap = tf.square(y_pred[:,2]-y_actual[:,2])
    sin_gap =  tf.square(y_pred[:,3]-y_actual[:,3])

    loss = 1.2*x_gap + 1.2*y_gap + cos_gap + sin_gap

    #return tf.math.sqrt(tf.math.reduce_mean(loss))
    return tf.math.reduce_mean(loss)

def main():
    rospy.init_node('get_d435i_module_image', anonymous=True)
    listener_rgb = Get_image()
    listener_depth = Get_image()
    reload_sm_keras = tf.keras.models.load_model(net_path,custom_objects={'custom_loss': custom_loss})
    reload_sm_keras.summary()

    while not rospy.is_shutdown():
        listener_depth.display_mode = 'depth'
        listener_rgb.display_mode = 'rgb'
        if(listener_rgb.display_mode == 'rgb')and(type(listener_rgb.cv_image) is np.ndarray):
            rgb_img = listener_rgb.cv_image
            #cv2.imshow("rgb module image", listener_rgb.cv_image)
        if(listener_depth.display_mode == 'depth')and(type(listener_depth.cv_depth) is np.ndarray):
            depth_img=cv2.cvtColor(listener_depth.cv_depth,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("depth module image", depth_img)
        ####################################
        (h, w) = depth_img.shape[:2]
        for i in range(h):
            for j in range(w):
                if depth_img[i][j]<=80:
                    depth_img[i][j]=rand.randint(243,245)
        kernel = np.ones((3,3), np.uint8)
        depth_img = cv2.dilate(depth_img, kernel, iterations = 1)
        #####Dilation 
        kernel = np.ones((3,3), np.uint8)
        depth_img = cv2.erode(depth_img, kernel, iterations = 1)
        ####################################
        net_input = depth_img.reshape(-1,480,640,1)/255
        predict_point=reload_sm_keras.predict([net_input])
        degree = arctan_recovery(predict_point[0][2],predict_point[0][3])
        predict_point[0][0]=predict_point[0][0]*640
        predict_point[0][1]=predict_point[0][1]*480
        temp_x,temp_y=trans_degree(predict_point[0][0],predict_point[0][1],degree)
        rgb_img = plot_result(rgb_img,predict_point[0][0],predict_point[0][1],temp_x,temp_y)
        depth_img = plot_result(depth_img,predict_point[0][0],predict_point[0][1],temp_x,temp_y)
        cv2.imshow("rgb module image",rgb_img)
        cv2.imshow("depth module image", depth_img)

            # key_num = cv2.waitKey(10)
            # print(key_num)
            # if key_num == 115:
            #     cv2.imwrite(path_depth+str(pic_num) + '.jpg',listener.cv_depth)
            #     print('Save picture '+str(pic_num))
            #     pic_num=pic_num+1
        


        cv2.waitKey(1)

if __name__ == "__main__":
    main()
    