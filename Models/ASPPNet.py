import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.backend as K
from keras import optimizers
from keras import activations
from Metrics import *

def Dense_Block(inputs, filter_size):
    #BN=>RELU=>CONV=>DROPOUT
    inputs = BatchNormalization()(inputs)
    inputs = Activation('relu')(inputs)
    op1 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op1 = Dropout(0.2)(op1)
    op1_c = concatenate([inputs, op1], axis = 3)

        
    #BN=>RELU=>CONV=>DROPOUT
    op2 = BatchNormalization()(op1_c)
    op2 = Activation('relu')(op2)
    op2 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op2)
    op2 = Dropout(0.2)(op2)
    op2_c = concatenate([op1_c, op2], axis = 3)
        
    #BN=>RELU=>CONV=>DROPOUT
    op3 = BatchNormalization()(op2_c)
    op3 = Activation('relu')(op3)
    op3 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op3)
    op3 = Dropout(0.2)(op3)
    op3_c = concatenate([op2_c, op3], axis = 3)

    #BN=>RELU=>CONV=>DROPOUT
    op4 = BatchNormalization()(op3_c)
    op4 = Activation('relu')(op4)
    op4 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op4)
    op4 = Dropout(0.2)(op4)
    op4_c = concatenate([op1, op2, op3], axis = 3)

    return op4_c

def FC_Transition_Down(inputs, filter_size):
    #BN=>RELU=>CONV=>DROPOUT=>MAXPOOLING
    inputs = BatchNormalization()(inputs)
    inputs = Activation('relu')(inputs)
    op1 = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op1 = Dropout(0.2)(op1)
    op1 = MaxPooling2D(pool_size=(2, 2))(op1)
    return op1 

def ASPP_Transition_Down(inputs, filter_size):
    #BN=>RELU=>CONV=>DROPOUT=>MAXPOOLING
    inputs = BatchNormalization()(inputs)
    inputs = Activation('relu')(inputs)
    op1 = Conv2D(filter_size, 3, strides=2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op1 = Dropout(0.2)(op1)
    return op1
        
def Transition_Up(inputs, filter_size):
    #3*3 Transposed Convolution with stride = 2
    op1 = Conv2DTranspose(filter_size, (3,3), strides=(2,2), padding='same')(inputs)
    op1 = Activation('relu')(op1)
    return op1 

def Pyramid_Pool(inputs, filter_size):
    op1 = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op2 = Conv2D(filter_size, 3, dilation_rate= 6,activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op3 = Conv2D(filter_size, 3, dilation_rate = 12,activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    op4 = Conv2D(filter_size, 3, dilation_rate = 18,activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    merge = concatenate([op1,op2,op3,op4,inputs], axis = 3)
    merge = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge)
    return merge

def asppnet(pretrained_weights = None, input_size = (256,256,1)):

    inputs = Input(input_size)

    #3*3 Conv
    conv = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)

    # 4 dense blocks supported by Transition Down Layers
    added1 = Dense_Block(inputs, 64)
    pool1 = ASPP_Transition_Down(added1, 64)

    added2 = Dense_Block(pool1, 128)
    pool2 = ASPP_Transition_Down(added2, 128)
    
    added3 = Dense_Block(pool2, 256)
    pool3 = ASPP_Transition_Down(added3, 256)
    
    added4 = Dense_Block(pool3, 512)
    pool4 = ASPP_Transition_Down(pool3, 512)

    pooled_vector = Pyramid_Pool(pool4, 512)

    trans1 = Transition_Up(pooled_vector, 512)
    merge6 = concatenate([trans1, added4], axis = 3)
    
    db1 = Dense_Block(merge6,512)
    trans2 = Transition_Up(db1, 512)
    merge7 = concatenate([trans2, added3], axis = 3)
    
    db2 = Dense_Block(merge7,256)
    trans3 = Transition_Up(db2, 256)
    merge8 = concatenate([trans3, added2], axis = 3)

    db3 = Dense_Block(merge8,128)
    trans4 = Transition_Up(db3, 128)
    merge9 = concatenate([trans4, added1], axis = 3)

    db3 = Dense_Block(merge9,64)
    final = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(db3)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(final)
    model = Model(input = inputs, output = conv10)
    dl = balanced_cross_entropy(0.8)
    model.compile(optimizer = SGD(lr=5e-6, decay=1e-6, momentum=0.5, nesterov=True), loss = dl, metrics = ['accuracy', dice_score])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

