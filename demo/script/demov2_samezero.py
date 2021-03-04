#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
sys.path.insert(1, "/home/allen/.local/lib/python3.5/site-packages/")
sys.path.insert(2, "/home/allen/realsensepkg/catkin_workspace/install/lib/python3/dist-packages")
import rospy
import cv2
from get_rs_image import Get_image,Get_imagev2
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
from demo.srv import lungrasp 
psnr_limit = 46.6

target_confi =False
target_x = 0.0
target_y = 0.0
target_z = 0.0
target_ang = 0.0
target_is_done = True
typeof = 'nonotrain'
typeof = 'notrain'
save_imgpath = '/home/allen/dl_grasp/src/result/nono_trained/'
save_imgpath = '/home/allen/dl_grasp/src/result/no_trained/'
net_path = '/home/allen/dl_grasp/src/train/Save_net/14object/drop/0106_samezero'
def cal_psnr(im1,im2):
    global target_confi
    im1= im1.reshape(480,640,1)
    im2= im2.reshape(480,640,1)
    #im1 = tf.image.convert_image_dtype(im1, tf.float32)
    #im2 = tf.image.convert_image_dtype(im2, tf.float32)
    y = tf.image.psnr(im1, im2, max_val=255)
    num = float(y)
    if num<psnr_limit:
        target_confi = True
    else:
        target_confi = False
    return num

def get_depth(img,x,y):
    global target_z
    range_size = 5
    (x,y)=(int(x),int(y))
    b = img[y-5:y+5,x-5:x+5]
    avg_depth = float(np.mean(b))
    target_z=22*(avg_depth+243)/243
    #print ('target_z',target_z)
    #target_z = float(avg_depth)
    return target_z

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
    global target_ang
    predict_degree = 0.5*np.arctan2(sin_x,cos_x)
    predict_degree=predict_degree/math.pi*180
    target_ang = predict_degree
    return predict_degree


def custom_loss(y_actual,y_pred):
    x_gap = tf.square(y_pred[:,0]-y_actual[:,0])
    y_gap = tf.square(y_pred[:,1]-y_actual[:,1])
    cos_gap = tf.square(y_pred[:,2]-y_actual[:,2])
    sin_gap =  tf.square(y_pred[:,3]-y_actual[:,3])

    loss = 1.2*x_gap + 1.2*y_gap + cos_gap + sin_gap

    #return tf.math.sqrt(tf.math.reduce_mean(loss))
    return tf.math.reduce_mean(loss)
def trans_inf(req):
    print('Get the request')
    # global target_confi
    # global target_x
    # global target_y
    # global target_z
    # global target_ang
    # global target_is_done
    a = target_confi
    b = target_x
    c = target_y
    d = target_z
    e = target_ang
    f = target_is_done
    return [a,b,c,d,e,f]
def main():
    rospy.init_node('get_d435i_module_image', anonymous=True)
    #s=rospy.Service('grasp_detection',lungrasp,trans_inf)
    listener_rgb = Get_image()
    listener_depth = Get_imagev2()
    reload_sm_keras = tf.keras.models.load_model(net_path,custom_objects={'custom_loss': custom_loss})
    reload_sm_keras.summary()
    background = cv2.imread("/home/allen/dl_grasp/src/data_expend/background/background_7.jpg",0)
    s=rospy.Service('grasp_detection',lungrasp,trans_inf)
    #rospy.spin()
    while not rospy.is_shutdown():
        listener_depth.display_mode = 'depth'
        listener_rgb.display_mode = 'rgb'
        if(listener_rgb.display_mode == 'rgb')and(type(listener_rgb.cv_image) is np.ndarray):
            rgb_img = listener_rgb.cv_image
        if(listener_depth.display_mode == 'depth')and(type(listener_depth.cv_depth) is np.ndarray):
            depth_img=listener_depth.cv_depth
        ####################################
        # (h, w) = depth_img.shape[:2]
        # for i in range(h):
        #     for j in range(w):
        #         if depth_img[i][j]<=80:
        #             depth_img[i][j]=rand.randint(243,245)
        # kernel = np.ones((3,3), np.uint8)
        # depth_img = cv2.dilate(depth_img, kernel, iterations = 1)
        # #####Dilation 
        # kernel = np.ones((3,3), np.uint8)
        # depth_img = cv2.erode(depth_img, kernel, iterations = 1)
        ####################################
        psnr = cal_psnr(background,depth_img)
        print('psnr : ',psnr)
        net_input = depth_img.reshape(-1,480,640,1)/255
        predict_point=reload_sm_keras.predict([net_input])
        degree = arctan_recovery(predict_point[0][2],predict_point[0][3])
        predict_point[0][0]=predict_point[0][0]*320+320
        predict_point[0][1]=predict_point[0][1]*240+240
        global target_x
        global target_y
        target_x = predict_point[0][0]
        target_y = predict_point[0][1]
        temp_x,temp_y=trans_degree(predict_point[0][0],predict_point[0][1],degree)
    
        avg_depth = get_depth(depth_img,predict_point[0][0],predict_point[0][1])
        #print('avg_depth : ',avg_depth)
        if psnr<psnr_limit:
            rgb_img = plot_result(rgb_img,predict_point[0][0],predict_point[0][1],temp_x,temp_y)
            depth_img = plot_result(depth_img,predict_point[0][0],predict_point[0][1],temp_x,temp_y)
        cv2.imshow("rgb module image",rgb_img)
        cv2.imshow("depth module image", depth_img)
        
        ##############################
        dirs = os.listdir(save_imgpath)
        img_num = len(dirs)/2
        key_num = cv2.waitKey(1)
            #print(key_num)
        if key_num == 115:
            save_num = img_num+1
            cv2.imwrite(save_imgpath+str(typeof)+str(int(save_num))+'_rgb.jpg',rgb_img)
            cv2.imwrite(save_imgpath+str(typeof)+str(int(save_num))+'_dep.jpg',depth_img)
            print('Save picture '+str(int(save_num)))
        ################################
        #cv2.waitKey(1)

if __name__ == "__main__":
    main()
    