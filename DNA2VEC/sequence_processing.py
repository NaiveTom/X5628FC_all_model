import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 全部使用嵌套结构



# 句子拆成单词
def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq



# 句子拆成单个碱基
def sentence2char(str_set):
    char_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)):
            if('N' in sr[i]):
                tmp.append('null')
            else:
                tmp.append(sr[i])
        char_seq.append(' '.join(tmp))
    return char_seq



# 单词转化为标号
def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq



# 单个碱基转化成标号
def char2num(charseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(charseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq



# 句子转化成标号
def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq



# speid 就是完全离散
def sentence2num_speid(str_set, tokenizer, MAX_LEN):
    charseq = sentence2char(str_set)
    numseq = char2num(charseq, tokenizer, MAX_LEN)
    return numseq



# 对应的 ACGT 代号
def get_tokenizer():
    f = ['A', 'C', 'G', 'T']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer



# 对应的 ACGT 代号
def get_tokenizer_speid():
    f = ['A', 'C', 'G', 'T']
    res = []
    for i in f:
        res.append(i)
    res = np.array(res)
    NB_WORDS = 5
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer



# 获取数据
def get_data(enhancers, promoters):
    tokenizer = get_tokenizer()
    MAX_LEN = 3000
    X_en = sentence2num(enhancers, tokenizer, MAX_LEN)
    MAX_LEN = 2000
    X_pr = sentence2num(promoters, tokenizer, MAX_LEN)

    return X_en, X_pr



# 获取数据
def get_data_speid(enhancers, promoters):
    tokenizer = get_tokenizer_speid()
    MAX_LEN = 3000
    X_en = sentence2num_speid(enhancers, tokenizer, MAX_LEN)
    MAX_LEN = 2000
    X_pr = sentence2num_speid(promoters, tokenizer, MAX_LEN)

    return X_en, X_pr



# In[ ]:



'''
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
name = names[5] # 使用 'NHEK' 细胞群
train_dir = '/home/ycm/data/%s/train/' % name
imbltrain = '/home/ycm/data/%s/imbltrain/' % name
test_dir = '/home/ycm/data/%s/test/' % name
Data_dir = '/home/ycm/data/%s/' % name
print('Experiment on %s dataset' % name)
'''

'''
print('Loading seq data...')
enhancers_tra = open(train_dir+'%s_enhancers.fasta' %
                     name, 'r').read().splitlines()[1::2]
promoters_tra = open(train_dir+'%s_promoters.fasta' %
                     name, 'r').read().splitlines()[1::2]
y_tra = np.loadtxt(train_dir+'%slabels.txt' % name)
'''


####################
# anchor1
####################

from fancy_print import fancy_print

fancy_print('Loading X5628FC anchor1 ...')

f = open('../anchor_data/seq.anchor1.pos.txt', 'r')
anchor1 = f.readlines()
len_pos = len(anchor1)
fancy_print('len_pos', len_pos, '=')

f = open('../anchor_data/seq.anchor1.neg2.txt', 'r')
anchor1.extend(f.readlines())
len_neg = len(anchor1) - len_pos
fancy_print('len_neg', len_neg, '=')

label = [1] * len_pos + [0] * len_neg



from sklearn.model_selection import train_test_split

anchor1_train, anchor1_test, anchor1_label_train, anchor1_label_test = train_test_split(anchor1, label, test_size = 0.1, random_state = 42)

fancy_print('anchor1_train.shape', np.array(anchor1_train).shape, '+')
fancy_print('anchor1_test.shape', np.array(anchor1_test).shape, '+')
fancy_print('anchor1_label_train.shape', np.array(anchor1_label_train).shape, '+')
fancy_print('anchor1_label_test.shape', np.array(anchor1_label_test).shape, '+')

####################
# anchor2
####################

fancy_print('Loading X5628FC anchor2 ...')

f = open('../anchor_data/seq.anchor2.pos.txt', 'r')
anchor2 = f.readlines()

f = open('../anchor_data/seq.anchor2.neg2.txt', 'r')
anchor2.extend(f.readlines())

anchor2_train, anchor2_test, anchor2_label_train, anchor2_label_test = train_test_split(anchor2, label, test_size = 0.1, random_state = 42)



fancy_print('X5628FC anchor2 finished ...')

'''
im_enhancers_tra = open(imbltrain+'%s_enhancers.fasta' %
                        name, 'r').read().splitlines()[1::2]
im_promoters_tra = open(imbltrain+'%s_promoters.fasta' %
                        name, 'r').read().splitlines()[1::2]
y_imtra = np.loadtxt(imbltrain+'%slabels.txt' % name)
'''

'''
enhancers_tes = open(test_dir+'%s_enhancers.fasta' %
                     name, 'r').read().splitlines()[1::2]
promoters_tes = open(test_dir+'%s_promoters.fasta' %
                     name, 'r').read().splitlines()[1::2]
y_tes = np.loadtxt(test_dir+'%slabels.txt' % name)
'''
'''
print('平衡训练集')
print('pos_samples:'+str(int(sum(y_tra))))
print('neg_samples:'+str(len(y_tra)-int(sum(y_tra))))
print('不平衡训练集')
# print('pos_samples:'+str(int(sum(y_imtra))))
# print('neg_samples:'+str(len(y_imtra)-int(sum(y_imtra))))
print('测试集')
print('pos_samples:'+str(int(sum(y_tes))))
print('neg_samples:'+str(len(y_tes)-int(sum(y_tes))))
'''


# In[ ]:
#X_en_speid,X_pr_speid = get_data_speid(enhancers_tra,promoters_tra)
# np.savez(Data_dir+'%s_train_speid.npz'%name,X_en_tra_speid=X_en_speid,X_pr_tra_speid=X_pr_speid,y_tra_speid=y_tra)
# X_en_tes_speid,X_pr_tes_speid=get_data_speid(enhancers_tes,promoters_tes)
# np.savez(Data_dir+'%s_test_speid.npz'%name,X_en_tes_speid=X_en_tes_speid,X_pr_tes_speid=X_pr_tes_speid,y_tes_speid=y_tes)
'''
X_en_tra, X_pr_tra = get_data(enhancers_tra, promoters_tra)
X_en_imtra, X_pr_imtra = get_data(im_enhancers_tra, im_promoters_tra)
X_en_tes, X_pr_tes = get_data(enhancers_tes, promoters_tes)
'''
####################
# 填充数据
####################

fancy_print('train enconding ...')

X_enhancer_train, X_promoter_train = get_data(anchor1_train, anchor2_train)

fancy_print('test enconding ...')

X_enhancer_test, X_promoter_test = get_data(anchor1_test, anchor2_test)

fancy_print('writing data into .npz ...')


'''
np.savez(Data_dir+'%s_train.npz' %
         name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
np.savez(Data_dir+'im_%s_train.npz' %
         name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
np.savez(Data_dir+'%s_test.npz' %
         name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)
'''
####################
# 写入数据
####################

name = 'X5628FC'

np.savez('%s_train.npz' %
         name, X_en_tra = X_enhancer_train, X_pr_tra = X_promoter_train, y_tra = anchor1_label_train)

np.savez('%s_test.npz' %
         name, X_en_tes = X_enhancer_test, X_pr_tes = X_promoter_test, y_tes = anchor1_label_test)
