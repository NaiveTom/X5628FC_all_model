# 垃圾回收模块
import gc


from model import get_model
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import numpy as np
# np.set_printoptions(threshold=np.inf)

import os
# 使用CPU进行处理（超级计算机）
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 禁止显示系统信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


'''
import tensorflow as tf
# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) 
    except RuntimeError as e:
        # 打印异常
        print(e)
'''


'''
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
for name in names:
'''

# 前面没有垃圾需要回收
# print('gc.collect() = ', end='')
# print(gc.collect())


for epoch in range(1000):

    name = 'X5628FC'

    # from fancy_print import fancy_print
    # fancy_print('加载开始', None, '+')

    model = get_model()
    model.load_weights("./model/%sModel%s.h5" % (name, epoch))

    # fancy_print('加载成功！', None, '=')

    # Data_dir = '/home/ycm/data/%s/' % name
    test = np.load('%s_test.npz' % name)
    X_en_tes, X_pr_tes, y_test = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
    # fancy_print('y_test', y_test[:100], '=')


    """
    print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
    """
    '''
    y_pred = model.predict([X_en_tes, X_pr_tes])
    auc = roc_auc_score(y_tes, y_pred)
    aupr = average_precision_score(y_tes, y_pred)
    f1 = f1_score(y_tes, np.round(y_pred.reshape(-1)))
    print("AUC : ", auc)
    print("AUPR : ", aupr)
    print("f1_score", f1)
    '''







    ##############################
    #
    # prediction
    #
    ##############################

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import cross_val_score
    from scipy import interp

    import os
    import matplotlib.pyplot as plt

    # fancy_print('cross_val_score(clf, x_test, y_test, cv=5)', cross_val_score(clf, x_test, y_test, cv=5))

    # 利用model.predict获取测试集的预测值
    y_score = model.predict([X_en_tes, X_pr_tes])
    # fancy_print('y_score', y_score, '.')



    # 计算F1
    from sklearn.metrics import f1_score
    y_pred = y_score.argmax(axis=-1)
    # fancy_print('f1', f1_score(y_test, y_pred, average='binary'), '.')

    

    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 二进制化输出
    # 转成独热码
    y_test = y_test.tolist()
    y_temp_1 = []; y_temp_0 = []
    for each in y_test:
        if each == 1: y_temp_1.append(1); y_temp_0.append(0)
        else: y_temp_1.append(0); y_temp_0.append(1)
    y_test = [y_temp_0, y_temp_1]; y_test = np.transpose(y_test)
    # fancy_print('y_test', y_test, '.')
    # fancy_print('y_test.shape', y_test.shape)

    # y_test = label_binarize(y_test, classes=[0, 1, 2])  # 二分类
    # fancy_print('y_test', y_test)

    # 二进制化输出
    # 转成独热码，不需要了，这次直接就是两组
    
    y_score = y_score.tolist()
    y_temp_1 = []; y_temp_0 = []
    for each in y_score:
        y_temp_1.append(float(each[0]))
        y_temp_0.append(1-float(each[0]))
    y_score = [y_temp_0, y_temp_1]; y_score = np.transpose(y_score)
    # fancy_print('y_score', y_score)
    # fancy_print('y_score.shape', y_score.shape)
    
    # y_score = label_binarize(y_score, classes=[0, 1, 2])  # 二分类
    # fancy_print('y_score', y_score)



    n_classes = y_test.shape[1] # n_classes = 2
    # fancy_print('n_classes', n_classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fancy_print('fpr', fpr)
    # fancy_print('tpr', tpr)
    # fancy_print('roc_auc', roc_auc)

    

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(ROC)')
    plt.legend(loc="lower right")

    plt.savefig('./aucroc/epoch_'+str(epoch)+'_auc.png')
    plt.close()
    # plt.show()



    # 经常收集垃圾，不然会崩溃的，内存不够用
    print('epoch = ', end='')
    print(epoch, end='\t')
    print('gc.collect() = ', end='')
    print(gc.collect())