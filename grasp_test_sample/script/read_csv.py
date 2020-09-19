#!/usr/bin/env python3
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
file_path = "/home/allen/dl_grasp/src/grasp_test_sample/data/square_data.csv"

def read_csv(csvFile):
    with open(file_path, newline='') as csvFile:

        rows = csv.reader(csvFile)


        rows = csv.reader(csvFile, delimiter=',')

        for row in rows:
            print(row)
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
def create_result_array(data2,data3):
    result_array=np.zeros([len(data2),2],  dtype=float)
    print(type(result_array))
    for i in range(len(data2)):
        result_array[i]=[data2[i],data3[i]]
    return result_array
def create_photo_array(data1):
    
def main():
    data1, data2, data3 = pd_read_csv(file_path)
    result_array=create_result_array(data2,data3)
    print(result_array)

    
    
    
if __name__ == "__main__":
    main()