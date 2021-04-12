# 使用GPU模式，不然永远也训练不完
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import get_model
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split



'''
2021-04-11 16:53:06.007063: E tensorflow/stream_executor/dnn.cc:616] CUDNN_STATUS_INTERNAL_ERROR

in tensorflow/stream_executor/cuda/cuda_dnn.cc(2011): 'cudnnRNNBackwardData( cudnn.handle(), rnn_desc.handle(), 
model_dims.max_seq_length, output_desc.handles(), output_data.opaque(), output_desc.handles(), output_backprop_data.opaque(), 
output_h_desc.handle(), output_h_backprop_data.opaque(), output_c_desc.handle(), output_c_backprop_data.opaque(), 
rnn_desc.params_handle(), params.opaque(), input_h_desc.handle(), input_h_data.opaque(), input_c_desc.handle(), 
input_c_data.opaque(), input_desc.handles(), input_backprop_data->opaque(), input_h_desc.handle(), input_h_backprop_data->opaque(), 
input_c_desc.handle(), input_c_backprop_data->opaque(), workspace.opaque(), workspace.size(), reserve_space_data->opaque(), reserve_space_data->size())'

2021-04-11 16:53:06.007530: W tensorflow/core/framework/op_kernel.cc:1767] OP_REQUIRES failed at cudnn_rnn_ops.cc:1922: 
Internal: Failed to call ThenRnnBackward with model config: [rnn_mode, rnn_input_mode, rnn_direction_mode]: 3, 0, 0 , 
[num_layers, input_size, num_units, dir_count, max_seq_length, batch_size, cell_num_units]: [1, 64, 50, 1, 100, 32, 0] 

2021-04-11 16:53:06.007077: F tensorflow/stream_executor/cuda/cuda_dnn.cc:190] Check failed: status == CUDNN_STATUS_SUCCESS (7 vs. 0)Failed to set cuDNN stream.

解决方案
'''

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() # disable for tensorFlow V2

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



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



class roc_callback(Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        self.model.save_weights(
            "./model/{0}Model{1}.h5".format(self.name, epoch))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

#names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
# name=names[0]
# The data used here is the sequence processed by data_processing.py.

'''
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
for name in names:
'''
name = 'X5628FC'

# Data_dir = '/home/ycm/data/%s/' % name
train = np.load('%s_train.npz' % name)
X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']
model = get_model()
model.summary()
print('Traing %s cell line specific model ...' % name)



back = roc_callback(name=name)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 30, restore_best_weights = True)
# 绘图函数
plot_progress = PlotProgress(entity = ['loss', 'accuracy'])



history = model.fit([X_en_tra, X_pr_tra], y_tra, epochs=1000, batch_size=32, validation_split=0.11,
                    callbacks=[back, early_stopping, plot_progress])
t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')



model.save('dna2vec_best_model.h5')

print("开始时间:"+t1+"结束时间："+t2)