# -*- coding:utf-8 -*-

# 分割比例
# 0.8 : 0.1 : 0.1

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove unnecessary information

# 三代垃圾回收机制
import gc
gc.enable()

import numpy as np

# Print format
def fancy_print(n = None, c = None, s = '#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # Avoid confusion

# Set the GPU usage mode to be progressive to avoid full memory
# Get GPU list
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU to increase occupancy mode
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Set the GPU to increase occupancy mode')
    except RuntimeError as e:
        # print error(no GPU mode)
        fancy_print('RuntimeError', e)



##############################
#
# Build iterator
#
##############################

from keras.preprocessing.image import ImageDataGenerator

# 训练集：验证集：测试集 = 8：1：1
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.11) # set validation split

BATCH_SIZE = 32 # 一次大小

train_generator = train_datagen.flow_from_directory(directory = '../anchor_data/train/',
                                                    target_size = (20002, 5),
                                                    color_mode = 'grayscale',
                                                    class_mode = 'categorical',
                                                    batch_size = BATCH_SIZE,
                                                    subset = 'training',  # set as training data
                                                    shuffle = True, # must shuffle
                                                    seed = 42,
                                                    )
val_generator = train_datagen.flow_from_directory(directory = '../anchor_data/train/', # same directory as training data
                                                  target_size = (20002, 5),
                                                  color_mode = 'grayscale',
                                                  class_mode = 'categorical',
                                                  batch_size = BATCH_SIZE,
                                                  subset = 'validation', # set as validation data
                                                  shuffle = True, # must shuffle
                                                  seed = 42,
                                                  )



##############################
#
# loss数据可视化
#
##############################

import keras
from matplotlib import pyplot as plt



class PlotProgress(keras.callbacks.Callback):

    def __init__(self, entity = ['loss', 'accuracy']):
        self.entity = entity

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.accs = []
        self.val_accs = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        # 损失函数
        self.losses.append(logs.get('{}'.format(self.entity[0])))
        self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
        # 准确率
        self.accs.append(logs.get('{}'.format(self.entity[1])))
        self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

        self.i += 1

        # clear_output(wait=True)
        plt.figure(0)
        plt.clf() # 清理历史遗迹
        plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
        plt.legend()
        plt.savefig('loss.jpg')
        plt.pause(0.01)
        # plt.show()

        plt.figure(1)
        plt.clf()  # 清理历史遗迹
        plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
        plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
        plt.legend()
        plt.savefig('acc.jpg')
        plt.pause(0.01)
        # plt.show()



##############################
#
# Model building
#
##############################

from sklearn import metrics
from keras.callbacks import ModelCheckpoint

# Import X5628FC_model.py
# from X5628FC_model import *
from X5628FC_model_original import *

clf = model_def()

clf.summary() # Print model structure

from keras.optimizers import Adam
clf.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr = 0.0001, decay = 0.00001),
            metrics = ['accuracy'])

'''
filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
'''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 20, restore_best_weights = True)

# 绘图函数
plot_progress = PlotProgress(entity = ['loss', 'accuracy'])



gc.collect() # Recycle all generations of garbage to avoid memory leaks



"""
fancy_print('train_generator.next()[0][0]', train_generator.next()[0][0], '+')
fancy_print('train_generator.next()[1][0]', train_generator.next()[1][0], '+')
fancy_print('train_generator.next()[0].shape', train_generator.next()[0].shape, '+')
fancy_print('train_generator.next()[1].shape', train_generator.next()[1].shape, '+')
'''
train_generator.next()[0].shape = (32, 20002, 5, 1)
train_generator.next()[1].shape = (32, 2)
'''
fancy_print('val_generator.next()[0][0]', val_generator.next()[0][0], '-')
fancy_print('val_generator.next()[1][0]', val_generator.next()[1][0], '-')
fancy_print('val_generator.next()[0].shape', val_generator.next()[0].shape, '-')
fancy_print('val_generator.next()[1].shape', val_generator.next()[1].shape, '-')
'''
val_generator.next()[0].shape = (32, 20002, 5, 1)
val_generator.next()[1].shape = (32, 2)
'''
"""
##############################
#
# Model training
#
##############################

# No need to count how many epochs, keras can count
history = clf.fit_generator(generator = train_generator,
                            epochs = 300,
                            validation_data = val_generator,

                            steps_per_epoch = train_generator.samples // BATCH_SIZE,
                            validation_steps = val_generator.samples // BATCH_SIZE,

                            callbacks = [plot_progress, early_stopping],
                            )

clf.save('CNN_best_model.h5')
