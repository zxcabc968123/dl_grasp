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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
train_file_path = "/home/allen/dl_grasp/src/grasp_test_sample/data/square_data.csv"
test_file_path  = "/home/allen/dl_grasp/src/grasp_test_sample/data/square_data_test.csv"

export_path='/home/allen/dl_grasp/src/grasp_test_sample/SaveNet_200924'

test_image_path='/home/allen/dl_grasp/src/rs_d435i/pic/depth_train/locate/6.jpg'

def pd_read_csv(csvFile):
 
    data_frame = pd.read_csv(csvFile, sep=",")
    data1 = []
    data2 = []
    data3 = []
   
    for index in data_frame.index:
        data_row = data_frame.loc[index]
        data1.append(data_row["image_path"])
        data2.append(data_row["target_x"])
        data3.append(data_row["target_y"])

    return (data1,data2,data3)

def get_test_image(image_path,reload_sm_keras):
    image=cv2.imread(image_path,0)  
    image_data=image.reshape(-1,480,640,1)
    image_data=image_data/255
    result_point=reload_sm_keras.predict([image_data])
    cv2.circle(image, (result_point[0][0],result_point[0][1]),5,  (0, 0, 255),2)
    ###
    cv2.imshow('My Image', image)
    print(image)
    print('prdict locate : x: {}  y: {}'.format(result_point[0][0],result_point[0][1]))
    cv2.waitKey(0)
def main():
    data1, data2, data3 = pd_read_csv(train_file_path)
    test_data1, test_data2, test_data3 = pd_read_csv(test_file_path)
    reload_sm_keras = tf.keras.models.load_model(export_path)
    reload_sm_keras.summary()
    for i in range(10):
        get_test_image(data1[i],reload_sm_keras)
    #for i in range(5):
        #get_test_image(test_data1[i],reload_sm_keras)
    #get_test_image(data1[9],reload_sm_keras)
    
    
if __name__ == "__main__":
    main()