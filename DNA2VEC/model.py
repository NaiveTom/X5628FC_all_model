from keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras
import numpy as np



MAX_LEN_en = 3000
MAX_LEN_pr = 2000
NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')

####################
# ACGT
####################
embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])



class AttLayer_1(Layer):
    
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer_1, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W_1')
        self.b = K.variable(self.init((self.attention_dim, )), name='b_1')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u_1')
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttLayer_1, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



class AttLayer_2(Layer):
    
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer_2, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W_2')
        self.b = K.variable(self.init((self.attention_dim, )), name='b_2')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='u_2')
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttLayer_2, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])





def get_model():

    ####################
    # 输入部分
    ####################
    enhancers = Input(shape=(MAX_LEN_en,))
    promoters = Input(shape=(MAX_LEN_pr,))

    ####################
    # embedding 部分
    ####################
    emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True)(enhancers)
    emb_pr = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True)(promoters)



    ####################
    # enhancer 输入部分
    ####################
    enhancer_conv_layer = Convolution1D(filters=64,
                                        kernel_size=60,
                                        padding='same',  # 'same'
                                        kernel_initializer='he_normal',
                                        )
    enhancer_max_pool_layer = MaxPooling1D(pool_size=30, strides=30)

    # Build enhancer branch
    enhancer_branch = Sequential()
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.5))
    enhancer_out = enhancer_branch(emb_en)



    ####################
    # promoter 输入部分
    ####################
    promoter_conv_layer = Convolution1D(filters=64,
                                        kernel_size=40,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        )
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)

    # promoter_length_slim = 2039
    # n_kernels_slim = 200
    # filter_length_slim = 20
    # Build promoter branch
    promoter_branch = Sequential()
    promoter_branch.add(promoter_conv_layer)
    promoter_branch.add(Activation("relu"))
    promoter_branch.add(promoter_max_pool_layer)
    promoter_branch.add(BatchNormalization())
    promoter_branch.add(Dropout(0.5))
    promoter_out = promoter_branch(emb_pr)

    # enhancer_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    # enhancer_max_pool_layer = MaxPooling1D(pool_size = 30, strides = 30)(enhancer_conv_layer)
    # promoter_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    # promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)

    ####################
    # 合并部分
    ####################
    l_gru_1 = Bidirectional(GRU(50, return_sequences=True), name='gru1')(enhancer_out)
    l_gru_2 = Bidirectional(GRU(50, return_sequences=True), name='gru2')(promoter_out)
    l_att_1 = AttLayer_1(50)(l_gru_1)
    l_att_2 = AttLayer_2(50)(l_gru_2)

    # 测试一下到底是因为什么崩溃的
    # l_att_1 = l_gru_1
    # l_att_2 = l_gru_2
    
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])

    merge_layer=Concatenate(axis=1)([l_att_1, l_att_2, subtract_layer, multiply_layer])
    # merge_layer = Concatenate(axis=1)([l_att_1, l_att_2])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    # l_gru = Bidirectional(LSTM(50))(dt)
    # l_att = AttLayer(50)(l_gru)
    # bn2 = BatchNormalization()(l_gru)
    # dt2 = Dropout(0.5)(bn2)
    # dt = BatchNormalization()(dt)
    # dt = Dropout(0.5)(dt)
    
    ####################
    # dense 部分
    ####################
    dt = Dense(512, kernel_initializer='glorot_uniform')(dt)
    dt = BatchNormalization()(dt)
    dt = Activation('relu')(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([enhancers, promoters], preds)
    adam = keras.optimizers.Adam(lr=5e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    
    return model
