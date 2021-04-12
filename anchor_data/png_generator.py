# -*- coding:utf-8 -*-

import numpy as np
import os

import gc # Garbage collection mechanism
gc.enable()

# Number of sampling(use all will increase the burden of the computer)
# Use all: SAMPLE = None
# Use 5000: SAMPLE = 5000
SAMPLE = None

# fancy_print for checking
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # Avoid confusion



##############################
#
# Read genetic data and add spaces for later processing
#
##############################

def read_data(name, file_dir): # name is for printing

    # Read data
    f = open(file_dir, 'r')
    data = f.readlines()

    data = data[ : SAMPLE ] # Divide a smaller size for testing, if it is None, then select all

    # Replace with a split format, and remove the line break
    for num in range( len(data) ):
        data[num] = data[num].replace('A', '0 ').replace('C', '1 ').replace('G', '2 ') \
                    .replace('T', '3 ').replace('N', '4 ').replace('\n', '')

    f.close()
        
    fancy_print(name + '.shape', np.array(data).shape, '=')

    return np.array(data)



##############################
#
# Split data set into test set, validation set, training set
#
##############################

def data_split(name, data):

    train_split_rate = 0.9 # 0.8 : 0.1 : 0.1
    
    print('-' * 40); print(name); print()

    print('train_split_rate', train_split_rate)
    print('test_split_rate', 1 - train_split_rate)
    
    print()
    
    import math
    
    length = math.floor( len(data) ) # Get the length
    train = data[ : int(length * train_split_rate) ]
    test = data[ int(length * train_split_rate) : ]

    print('len(train_set)', len(train))
    print('len(test_set)', len(test))
    
    print('-' * 40); print()
    
    return train, test



##############################
#
# onehot enconding
#
##############################

# The first parameter is the data to be encoded, and the second parameter is OneHotEncoder
# ACGTN是类别数
def onehot_func(data, ACGTN):

    from keras.utils import to_categorical

    data_onehot = []
    for i in range(len(data)):
        data_onehot.append( np.transpose(to_categorical(data[i].split(), ACGTN)) )

    data_onehot = np.array(data_onehot)

    return data_onehot





##############################
#
# Call function for information processing
#
##############################

def data_process():

    ####################
    # Read genetic data
    ####################

    anchor1_pos_raw = read_data('anchor1_pos', './seq.anchor1.pos.txt')
    anchor1_neg2_raw = read_data('anchor1_neg2', './seq.anchor1.neg2.txt')
    anchor2_pos_raw = read_data('anchor2_pos', './seq.anchor2.pos.txt')
    anchor2_neg2_raw = read_data('anchor2_neg2', './seq.anchor2.neg2.txt')

    gc.collect()  # Recycle all generations of garbage to avoid memory leaks



    ####################
    # shuffle 数据
    ####################
    
    if SAMPLE == None: # 全部混洗
        index = np.random.choice(anchor1_pos_raw.shape[0], size = anchor1_pos_raw.shape[0], replace = False)
        fancy_print('index_size = anchor1_pos_raw.shape[0]', anchor1_pos_raw.shape[0])
    else: # 混洗一部分，后面的没了，提高效率
        index = np.random.choice(SAMPLE, size = SAMPLE, replace = False)
        fancy_print('index_size = SAMPLE', SAMPLE)

    anchor1_pos = anchor1_pos_raw[index]
    anchor2_pos = anchor2_pos_raw[index]
    anchor1_neg2 = anchor1_neg2_raw[index]
    anchor2_neg2 = anchor2_neg2_raw[index]

    gc.collect()  # Recycle all generations of garbage to avoid memory leaks
    


    ####################
    # Call function for split processing
    ####################

    anchor1_pos_train, anchor1_pos_test = data_split('anchor1_pos', anchor1_pos)
    anchor1_neg2_train, anchor1_neg2_test = data_split('anchor1_neg2', anchor1_neg2)
    anchor2_pos_train, anchor2_pos_test = data_split('anchor2_pos', anchor2_pos)
    anchor2_neg2_train, anchor2_neg2_test = data_split('anchor2_neg2', anchor2_neg2)

    gc.collect()  # Recycle all generations of garbage to avoid memory leaks





    ##############################
    #
    # Write picture
    #
    ##############################

    # Write picture module
    # pip install imageio
    import imageio
    from skimage import img_as_ubyte

    # Convert to onehot encoding
    from keras.utils import to_categorical

    ACGTN = 5 # 类别数量

    fancy_print('one-hot enconding',
                '[ [A], [C], [G], [T], [N] ]\n' + str(to_categorical(['0', '1', '2', '3', '4'], ACGTN)))

    ####################
    # Save the training set as a picture
    ####################

    LEN_PER_LOAD = 1000 # The smaller the faster, 1000 is just right

    pic_num = 0

    # Here it is processed in blocks, processing 1000 at a time, because onehot is a second-order complexity, don’t make it too big
    for i in range( int(len(anchor1_pos_train)/LEN_PER_LOAD)+1 ):

        # Show the percentage
        print('\nThe training set and labels are being stored in blocks, block number =', str(i), '/', int( len(anchor1_pos_train)/LEN_PER_LOAD) )

        if (i+1)*LEN_PER_LOAD > len(anchor1_pos_train): # This code deals with the little tail(ending block) problem

            try: # Maybe the little tail(ending block) is 0
                anchor1_pos_train_onehot = onehot_func( anchor1_pos_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor1_neg2_train_onehot = onehot_func( anchor1_neg2_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor2_pos_train_onehot = onehot_func( anchor2_pos_train[ i*LEN_PER_LOAD : ], ACGTN )
                anchor2_neg2_train_onehot = onehot_func( anchor2_neg2_train[ i*LEN_PER_LOAD : ], ACGTN )
            except:
                print('The size of the last block is 0, this block has been skipped (to avoid errors)')

        else: # This code handles the normal blocking split
            
            anchor1_pos_train_onehot = onehot_func( anchor1_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor1_neg2_train_onehot = onehot_func( anchor1_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor2_pos_train_onehot = onehot_func( anchor2_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )
            anchor2_neg2_train_onehot = onehot_func( anchor2_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ACGTN )

        # combined together
        train_pos = np.dstack((anchor1_pos_train_onehot, anchor2_pos_train_onehot)) # Positive & Positive Merge horizontally dstack & Merge vertically hstack
        train_neg = np.dstack((anchor1_neg2_train_onehot, anchor2_neg2_train_onehot)) # Negative & Negative Merge horizontally dstack & Merge vertically hstack

        # 检查用
        # fancy_print('anchor1_pos_train_onehot', anchor1_pos_train_onehot)

        print('Merged block size', train_pos.shape)
        print('PNG is being generated...')


        if train_pos.shape[0]==0 or train_pos.shape[1]==0 or train_pos.shape[2]==0:
            print('Invalid empty block, skipped!')
            continue # Empty block, skip the loop


        # pip install tqdm
        # progress bar
        import tqdm

        # Write pictures one by one
        for j in tqdm.trange( len(train_pos), ascii=True ):
            imageio.imwrite('./train/1/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_pos[j]))) # Must be transposed, because PNG is inverted
            imageio.imwrite('./train/0/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_neg[j]))) # Must be transposed, because PNG is inverted
            pic_num += 1
        
        gc.collect()  # Recycle all generations of garbage to avoid memory leaks



    ####################
    # Save the test set as picture
    ####################

    print('\n\nWriting test set and tags to png ...')

    anchor1_pos_test_onehot = onehot_func( anchor1_pos_test, ACGTN )
    anchor1_neg2_test_onehot = onehot_func( anchor1_neg2_test, ACGTN )
    anchor2_pos_test_onehot = onehot_func( anchor2_pos_test, ACGTN )
    anchor2_neg2_test_onehot = onehot_func( anchor2_neg2_test, ACGTN )

    # combined together
    test_pos = np.dstack((anchor1_pos_test_onehot, anchor2_pos_test_onehot)) # Positive & Positive Merge horizontally dstack & Merge vertically hstack
    test_neg = np.dstack((anchor1_neg2_test_onehot, anchor2_neg2_test_onehot)) # Negative & Negative Merge horizontally dstack & Merge vertically hstack

    print('Merged block size', test_pos.shape)
    print('PNG is being generated...')

    # Write pictures one by one
    for j in tqdm.trange( len(test_pos), ascii=True ):
        imageio.imwrite('./test/1/'+str(j)+'.png', img_as_ubyte(np.transpose(test_pos[j]))) # Must be transposed, because PNG is inverted
        imageio.imwrite('./test/0/'+str(j)+'.png', img_as_ubyte(np.transpose(test_neg[j]))) # Must be transposed, because PNG is inverted

    gc.collect()  # Recycle all generations of garbage to avoid memory leaks





########################################
#
# main function
#
########################################

if __name__ == '__main__':

    fancy_print('merge_before_train')

    # There are many benefits of using PNG: small size, fast speed, intuitive,
    # and you can directly use keras API (high efficiency)
    data_process()

    print('\nAll operations have been completed!')
    input('\nPlease press any key to continue...') # Avoid crash
