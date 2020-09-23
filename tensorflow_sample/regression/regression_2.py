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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([[-40,-30],  [14,24], [32,42], [46,56], [59,69], [72,82], [100,110]],  dtype=float)

EPOCHS=100
def main():
    for i,c in enumerate(celsius_q):
        print('{} is {}'.format(c,fahrenheit_a[i]))
    reg=keras.Sequential()
    reg.add(layers.Dense(1,input_shape=[1]))
    reg.add(layers.Dense(10,input_shape=[1]))
    reg.add(layers.Dense(10,activation='relu'))
    #reg.add(layers.Dense(2,activation='relu'))
    reg.add(layers.Dense(2))
    reg.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.006))
    #reg.compile(loss='mean_squared_error', optimizer='sgd')
    #show the network structure 
    reg.summary()
    result=reg.fit(celsius_q,fahrenheit_a,batch_size=1,epochs=EPOCHS)
    print('completely training')
    print(reg.get_weights())

    export_path='/home/allen/dl_grasp/src/tensorflow_sample/regression/SaveNet_2'
    reg.save(export_path, save_format='tf')
    print('Show input shape :', reg.input_shape)
    print(fahrenheit_a[0].shape)
    print('predict :{}'.format(reg.predict([0])))
    #################### Loss/epoch
    loss = result.history['loss']
    epochs_range = range(EPOCHS)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.savefig('./loss.png')
    plt.show()

if __name__ == "__main__":
    main()
