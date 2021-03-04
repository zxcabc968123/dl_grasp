#!/usr/bin/env python3
from my_classes_samezero import *
###################################################2211
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
#data_path = '/home/allen/dl_grasp/src/data_expend/expand_data/4object_2000-11-18_07_30_55.csv'
#test_path = '/home/allen/dl_grasp/src/data_expend/expand_data/4object_80_2020-11-18_07_50_16.csv'
#data_path = '/home/allen/dl_grasp/src/data_expend/expand_data/1000blackdata_2020-10-28_07_13_23_.csv'
#test_path = '/home/allen/dl_grasp/src/data_expend/expand_data/40data_2020-10-29_16_16_22_.csv'
data_path = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_2000_0126.csv'
test_path = '/home/allen/dl_grasp/src/data_expend/expand_data/5object_200_0126.csv'
#save_path = '/home/allen/dl_grasp/src/train/Save_net/14object/drop/1213_base'
save_path = '/home/allen/dl_grasp/src/train/Save_net/5object_lossnum_check/mse_num12'
model_type = 'Base mode(stride 2211)'
EPOCHS = 1000
batch_size = 50

training_generator = DataGenerator(data_path,batch_size)
test_generator = DataGenerator(test_path,batch_size)

custom_early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=50, 
    min_delta=0.001, 
    mode='min'
)

def custom_loss(y_actual,y_pred):
    x_gap = tf.square(y_pred[:,0]-y_actual[:,0])
    y_gap = tf.square(y_pred[:,1]-y_actual[:,1])
    cos_gap = tf.square(y_pred[:,2]-y_actual[:,2])
    sin_gap =  tf.square(y_pred[:,3]-y_actual[:,3])

    loss = 1.2*x_gap + 1.2*y_gap + cos_gap + sin_gap

    return tf.math.reduce_mean(loss)

def main():
    #solve_cudnn_error()
    CNN=keras.Sequential()
    #add convolution layer filter 32 3*3 activation funtion relu
    CNN.add(layers.Conv2D(16,(3,3),activation=lrelu,input_shape=(480,640,1),strides=(2,2)))
    #add pooling layer 3*3 478*638
    CNN.add(layers.MaxPooling2D((2,2)))
    #add convolution layer filter 16 3*3 activation funtion relu 239*319
    CNN.add(layers.Conv2D(16,(3,3),activation=lrelu,strides=(2,2)))
    #add pooling layer 3*3 237*317
    CNN.add(layers.MaxPooling2D((2,2)))
    # 119*159
    CNN.add(layers.Conv2D(32,(3,3),activation=lrelu))
    # 118*158
    CNN.add(layers.MaxPooling2D((2,2)))
    # 59*79
    ##
    CNN.add(layers.Conv2D(32,(3,3),activation=lrelu))
    # 58*78
    # CNN.add(layers.MaxPooling2D((2,2)))
    # 29*39
    ######################################
    #CNN.add(layers.Conv2D(64,(3,3),activation=lrelu))
    ######################################
    # 26*36
    #CNN.add(layers.MaxPooling2D((2,2)))

    CNN.add(layers.Flatten())
    
    #####Dropout
    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(64,activation=lrelu))

    #CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(64,activation=lrelu))

    CNN.add(layers.Dense(32,activation=lrelu))

    CNN.add(layers.Dense(32,activation=lrelu))
    #dropoutv5 add fully-connected layer
    #CNN.add(layers.Dense(64,activation=lrelu))

    #CNN.add(layers.Dense(4))
    CNN.add(layers.Dense(4))

    CNN.compile(loss='mean_square_error',optimizer=tf.keras.optimizers.Adam(0.0001),metrics=['mae'])
    #CNN.compile(loss=custom_loss,optimizer=tf.keras.optimizers.Adam(0.0001),metrics=['mae'])
    CNN.summary()

    #result=CNN.fit(train_photo_array,train_result_array,validation_data=(test_photo_array, test_result_array),batch_size=Batch_size,epochs=EPOCHS,shuffle=True,verbose=1,callbacks=[custom_early_stopping])
    result = CNN.fit_generator(training_generator,epochs=EPOCHS,verbose=1,validation_data=test_generator)
    print('save CNN to : ',save_path)
    CNN.save(save_path, save_format='tf')

    #################### Loss/epoch
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    acc = result.history['mae']
    if custom_early_stopping.stopped_epoch==0:
        epochs_range = range(EPOCHS)
    else:
        epochs_range = range(custom_early_stopping.stopped_epoch+1)
    
    plt.title(model_type)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Testing Loss',color=(100/255,255/255,100/255))
    #plt.plot(epochs_range, acc, label='Training MAE',color=(255/255,100/255,100/255))
    plt.legend(loc='upper right')
    plt.savefig(save_path+'/lossmae.png')
    plt.show()

if __name__ == "__main__":
    main()
    