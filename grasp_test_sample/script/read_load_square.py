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

export_path='/home/allen/dl_grasp/src/grasp_test_sample/SaveNet_200923'

test_image_path='/home/allen/dl_grasp/src/rs_d435i/pic/depth_train/locate/144.jpg'
def get_test_image(image_path,reload_sm_keras):
    image=cv2.imread(image_path,0)
    cv2.imshow('My Image', image)
    cv2.circle(image, (100,100),10, (0, 0, 255))
    image_data=image.reshape(-1,480,640,1)
    image_data=image_data/255
    result_point=reload_sm_keras.predict([image_data])
    print(type(result_point))
    print(result_point)
    print('prdict locate : x: {}  y: {}'.format(result_point[0][0],result_point[0][1]))
    cv2.waitKey(0)
def main():
    reload_sm_keras = tf.keras.models.load_model(export_path)
    reload_sm_keras.summary()
    get_test_image(test_image_path,reload_sm_keras)
if __name__ == "__main__":
    main()