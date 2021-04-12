# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

from keras import Sequential



##############################
#
# Model structure
#
##############################

def model_def():

    DROPOUT_RATE = 0.5 # Discard ratio
    KERNEL_SIZE = 24
  
    model = Sequential()


  
    model.add(Conv2D(64, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same', input_shape=((20002, 5, 1))))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))



    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    model.add(BatchNormalization())



    model.add(Conv2D(128, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, kernel_size=(KERNEL_SIZE, 1), strides=(4, 1), kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))



    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
  
    model.add(Flatten())
    model.add(BatchNormalization())
  
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(1000, activation='relu')) # 最后不加 Dropout
    model.add(Dense(2, activation='softmax')) # 输出层：softmax 归一化



    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':
    
    classifier = model_def()

    from keras.utils import plot_model
    plot_model(classifier, to_file = 'CNN_model.png')
