# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation

from keras import Sequential

from resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152



##############################
#
# Model structure
#
##############################

def model_def():

    model = resnet_18() # ResNet152网络，这样的话会训练的无比慢，所以建议用50或者101

    model.build( input_shape = (None, 20002, 5, 1) )
    
    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':
  
    classifier = model_def()

    from keras.utils import plot_model
    plot_model(classifier, to_file = 'RESNET_model.png')
