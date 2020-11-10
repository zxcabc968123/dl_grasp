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

train_file_path = '/home/allen/dl_grasp/src/data_expend/expand_data/1000blackdata_2020-10-28_07_13_23_.csv'
test_file_path = '/home/allen/dl_grasp/src/data_expend/expand_data/40data_2020-10-29_16_16_22_.csv'
save_path = '/home/allen/dl_grasp/src/train/Save_net/CNN_MSE201030'
batch_size = 5
EPOCHS = 200

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

def main():
    ####################   train  data_create
    data1, data2, data3, data4 = pd_read_csv(train_file_path)
    train_photo_array=create_photo_array(data1)
    train_result_array=create_result_array(data2,data3,data4)
    print('photo_array shape: {}'.format(train_photo_array.shape))
    print('result_array shape: {}'.format(train_result_array.shape))

    ####################   test   data_create
    data1, data2, data3, data4 = pd_read_csv(test_file_path)
    test_photo_array=create_photo_array(data1)
    test_result_array=create_result_array(data2,data3,data4)
    print('photo_array shape: {}'.format(test_photo_array.shape))
    print('result_array shape: {}'.format(test_result_array.shape))

    ###################   Network
    CNN=keras.Sequential()
    #add convolution layer filter 32 3*3 activation funtion relu
    CNN.add(layers.Conv2D(64,(2,2),activation='relu',input_shape=(480,640,1)))
    #add pooling layer 3*3 
    CNN.add(layers.MaxPooling2D((2,2)))
    #add convolution layer filter 16 3*3 activation funtion relu
    CNN.add(layers.Conv2D(32,(2,2),activation='relu'))
    #add pooling layer 3*3
    CNN.add(layers.MaxPooling2D((2,2)))

    CNN.add(layers.Conv2D(32,(2,2),activation='relu'))
    CNN.add(layers.Conv2D(32,(2,2),activation='relu'))
    CNN.add(layers.Conv2D(32,(2,2),activation='relu'))

    CNN.add(layers.MaxPooling2D((2,2)))

    CNN.add(layers.Flatten())
    #####Dropout
    CNN.add(layers.Dropout(0.1))

    CNN.add(layers.Dense(64,activation='relu'))
    CNN.add(layers.Dense(64,activation='relu'))

    CNN.add(layers.Dense(3))

    #CNN.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.0006),metrics=['mae'])
    CNN.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.0006),metrics=['mae'])
    CNN.summary()

    result=CNN.fit(train_photo_array,train_result_array,batch_size=1,epochs=EPOCHS)

    print('save CNN')
    CNN.save(save_path, save_format='tf')

    #################### Loss/epoch
    loss = result.history['loss']
    epochs_range = range(EPOCHS)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.savefig(save_path+'/loss.png')
    plt.show()
    #####
    #score = model.evaluate(x_test, y_test, verbose=0)

    
if __name__ == "__main__":
    main()
    

