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
  
    model = Sequential()


  
    model.add( Flatten( input_shape=((20002, 5, 1)) ) )
    model.add(BatchNormalization())
  
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(1000, activation='relu'))
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
    plot_model(classifier, to_file = 'dense_model.png')
