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
train_file_path = "/home/allen/dl_grasp/src/grasp_test_sample/data/square_data.csv"
test_file_path  = "/home/allen/dl_grasp/src/grasp_test_sample/data/square_data_test.csv"
EPOCHS=50000
def read_csv(csvFile):
    with open(train_file_path, newline='') as csvFile:

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
    for i in range(len(data2)):
        result_array[i]=[data2[i],data3[i]]
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

def main():
    ####################   train  data_create
    data1, data2, data3 = pd_read_csv(train_file_path)
    train_photo_array=create_photo_array(data1)
    train_result_array=create_result_array(data2,data3)
    print('photo_array shape: {}'.format(train_photo_array.shape))
    print('result_array shape: {}'.format(train_result_array.shape))
    
    ####################   test   data_create
    data1, data2, data3 = pd_read_csv(test_file_path)
    test_photo_array=create_photo_array(data1)
    test_result_array=create_result_array(data2,data3)
    print('photo_array shape: {}'.format(test_photo_array.shape))
    print('result_array shape: {}'.format(test_result_array.shape))

    ###################   Network
    CNN=keras.Sequential()
    #add convolution layer filter 32 3*3 activation funtion relu
    CNN.add(layers.Conv2D(32,(2,2),activation='relu',input_shape=(480,640,1)))
    #add pooling layer 3*3 
    CNN.add(layers.MaxPooling2D((2,2)))
    #add convolution layer filter 16 3*3 activation funtion relu
    CNN.add(layers.Conv2D(64,(2,2),activation='relu'))
    #add pooling layer 3*3
    CNN.add(layers.MaxPooling2D((2,2)))
    #Flat matrix to enter DNN network
    CNN.add(layers.Flatten())
    #deep layer*3
    CNN.add(layers.Dense(30,activation='relu'))
    CNN.add(layers.Dense(15,activation='relu'))
    CNN.add(layers.Dense(15,activation='relu'))
    #CNN.add(layers.Dense(10,activation='relu'))
    CNN.add(layers.Dense(2))
    #################### function define
    CNN.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.0006))
    #CNN.compile(loss='mean_squared_error', optimizer='sgd')
    #show the network structure 
    CNN.summary()
    ####################train
    result=CNN.fit(train_photo_array,train_result_array,batch_size=1,epochs=EPOCHS)
    #print(CNN.predict(test_photo_array[0]))
    ####################SaveNet
    print('save CNN')
    export_path='/home/allen/dl_grasp/src/grasp_test_sample/SaveNet_201006'
    CNN.save(export_path, save_format='tf')
    print(train_photo_array[0].shape)
    print(train_result_array[0])
    print(train_result_array[0].shape)
    print('Show input shape :', CNN.input_shape)
    
    #################### Loss/epoch
    loss = result.history['loss']
    epochs_range = range(EPOCHS)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.savefig('./loss.png')
    plt.savefig('/home/allen/dl_grasp/src/grasp_test_sample/SaveNet_201006/loss.png')
    plt.show()

if __name__ == "__main__":
    main()