import tensorflow as tf
from tensorflow import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], 
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import random


def show_predict(pict_num,x_test123,reload_sm_keras):
    a=x_test123[pict_num]
    print('before')
    print(a.shape)
    a=a.reshape(-1,28,28,1)
    print('after')
    print(a.shape)
    print('Show picture No. %d'%(pict_num+1))
    print(reload_sm_keras.predict(a))
    print('len is : ',reload_sm_keras.predict(a))
    print('predict shape: {}'.format(reload_sm_keras.predict(a).shape))
    print('predict_number :' ,np.argmax(reload_sm_keras.predict(a)))
    #For show image correctly with plt,use squeeze() to delete dimension with '1'
    a=a.squeeze()
    plt.imshow(a,cmap="binary")
    plt.show()
def main():
    export_path = '/home/allen/dl_grasp/src/tensorflow_sample/minist_number/SaveNet'
    #reload_sm_keras = keras.models.load_model(export_path)
    reload_sm_keras = tf.keras.models.load_model(export_path)
    reload_sm_keras.summary()
    (x_train,y_train),(x_test123,y_test123)=tf.keras.datasets.mnist.load_data()
    x_test123=x_test123/255
    x_test123=x_test123.reshape((10000,28,28,1))
    print('Show reload test data mean:',np.mean(reload_sm_keras.predict_classes(x_test123)==y_test123))
    print('Show reload input shape :',reload_sm_keras.input_shape)
    show_predict(random.randint(0,9999),x_test123,reload_sm_keras)
if __name__ == "__main__":

    main()
    