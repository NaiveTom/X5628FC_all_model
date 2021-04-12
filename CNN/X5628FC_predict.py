# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove unnecessary information

# Garbage collection mechanism
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

# Set the GPU usage mode to be progressive to avoid full memory at beginning
# Get GPU list
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU to increase occupancy mode
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Set the GPU to increase occupancy mode')
    except RuntimeError as e:
        # Printing error(No GPU mode)
        fancy_print('RuntimeError', e)



##############################
#
# Test set iterator
#
##############################

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255)

BATCH_SIZE = 32 # 一次大小

test_generator = datagen.flow_from_directory(directory = '../anchor_data/test/', target_size = (20002, 5),
                                             color_mode = 'grayscale',
                                             batch_size = BATCH_SIZE,
                                             shuffle = False) # 不需要 shuffle



##############################
#
# Load the pre-trained model
#
##############################

# Skip training and load the model directly
from keras.models import load_model
clf = load_model('CNN_best_model.h5')

gc.collect() # Recycle all generations of garbage to avoid memory leaks



##############################
#
# prediction
#
##############################

# New content added to evaluate model quality
# Calculate auc and draw roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Measure accuracy first
score = clf.evaluate_generator(generator = test_generator, steps = len(test_generator))
fancy_print('loss & acc', score)

# Print all content
# np.set_printoptions(threshold = np.inf)

# Use model.predict to get the predicted probability of the test set
y_prob = clf.predict_generator(generator = test_generator, steps = len(test_generator))
fancy_print('y_prob', y_prob, '.')
fancy_print('y_prob.shape', y_prob.shape, '-')



# Get label
label_test_tag = test_generator.class_indices
label_test_name = test_generator.filenames

# 没有反，这里是对的
label_test = []
for i in label_test_name:
    label = i.split('\\')[0] # Separate categories
    # 这里是对的
    label_test.append( int(label_test_tag[label]) ) # Make it into number

from keras.utils.np_utils import to_categorical
label_test = to_categorical(label_test)

fancy_print('label_test', label_test, '.')
fancy_print('label_test.shape', label_test.shape, '-')
gc.collect() # Recycle all generations of garbage to avoid memory leaks



# Calculate ROC curve and AUC for each category
fpr = dict()
tpr = dict()
roc_auc = dict()



# Two classification problem
n_classes = label_test.shape[1] # n_classes = 2
fancy_print('n_classes', n_classes) # n_classes = 2

# Draw ROC curve using actual category and predicted probability
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fancy_print('fpr', fpr)
fancy_print('tpr', tpr)
fancy_print('cnn_roc_auc', roc_auc)



plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color = 'darkorange', 
         lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc = "lower right")

plt.show()
