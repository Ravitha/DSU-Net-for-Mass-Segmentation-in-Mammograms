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

def conv1_block(inputs, filter_size):
        #1*1=>BN=>RELU
        op1 = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        op2 = BatchNormalization()(op1)
        op2 = Activation('relu')(op2)
        return op2

def conv2_block(inputs, filter_size):
        #CONV 3*3 + ReLU
        op1 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        op1 = Activation('relu')(op1)
        #CONV 3*3 + ReLU
        op1 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op1)
        op1 = Activation('relu')(op1)
        return op1

def Transpose_Block(inputs, filter_size):
        #TransposeConv 3*3 + ReLU
        op1 = Conv2DTranspose(filter_size, (3,3), strides=(2,2), padding='same')(inputs)
        op1 = Activation('relu')(op1)
        return op1 


def Dense_Block(inputs, filter_size):
        #BN=>RELU=>CONV
        inputs = BatchNormalization()(inputs)
        inputs = Activation('relu')(inputs)
        op1 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        
        #BN=>RELU=>CONV=>ADD(OP1)
        op2 = BatchNormalization()(op1)
        op2 = Activation('relu')(op2)
        op2 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op2)
        op2 = keras.layers.Add()([op1, op2])
        
        #BN=>RELU=>CONV=>ADD(OP1, OP2)
        op3 = BatchNormalization()(op2)
        op3 = Activation('relu')(op3)
        op3 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op3)
        op3 = keras.layers.Add()([op1, op2, op3])

        #BN=>RELU=>CONV=>ADD(OP1, OP2, OP3)
        op4 = BatchNormalization()(op3)
        op4 = Activation('relu')(op4)
        op4 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op4)
        op4 = keras.layers.Add()([op1, op2, op3, op4])

        #BN=>RELU=>CONV
        op5 = BatchNormalization()(op4)
        op5 = Activation('relu')(op5)
        op5 = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(op5)

        return op5

def attention_block(hlayer, llayer, filter_size):
        #CONV hlayer
        g = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(hlayer)
        #stide =1 CONV llayer
        xl = Conv2D(filter_size, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal', strides=2)(llayer)
        #Add
        added = keras.layers.Add()([g,xl])
        #relu=>conv=>sigmoid
        added = Activation('relu')(added)
        added = Conv2D(1, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(added)
        added = Activation('sigmoid')(added)
        #Upsample Added
        added = UpSampling2D(size = (2,2), interpolation='bilinear')(added)
        #Multiply llayer
        at = keras.layers.Multiply()([llayer,added])

        return at

        
def denseunet(pretrained_weights = None, input_size = (256,256,1)):
        inputs = Input(input_size)
        #CONV 3*3
        conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        
        #DENSE=>1*1=>Average Pooling
        fm_layer1 =Dense_Block(conv1,64)
        conv_layer1 = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(fm_layer1)        
        pool_tl1 = AveragePooling2D(pool_size=(2, 2))(conv_layer1)

        #DENSE=>1*1=>Average Pooling
        fm_layer2 =Dense_Block( pool_tl1 ,128)
        conv_layer2 = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(fm_layer2)        
        pool_tl2 = AveragePooling2D(pool_size=(2, 2))(conv_layer2)

        #DENSE=>1*1=>Average Pooling
        fm_layer3 =Dense_Block( pool_tl2 ,256)
        conv_layer3 = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(fm_layer3)        
        pool_tl3 = AveragePooling2D(pool_size=(2, 2))(conv_layer3)

        #DENSE=>1*1=>Average Pooling
        fm_layer4 =Dense_Block( pool_tl3 ,512)
        conv_layer4 = Conv2D(64, 1, activation = None, padding = 'same', kernel_initializer = 'he_normal')(fm_layer4)        
        pool_tl4 = AveragePooling2D(pool_size=(2, 2))(conv_layer4)

        #DENSE
        fm_layer5 = Dense_Block( pool_tl4 ,1024)

        clayer5= conv1_block(fm_layer5, 1024)
        at1 = attention_block(clayer5, fm_layer4, 16)
        T1 = Transpose_Block(fm_layer5, 512)
        c1 = concatenate([at1,T1], axis = 3)

        flayer4 = conv2_block(c1, 512)
        clayer4 = conv1_block(flayer4, 512)
        at2 = attention_block(clayer4, fm_layer3, 16)
        T2 = Transpose_Block(flayer4, 256)
        c2 = concatenate([at2, T2], axis = 3)

        flayer3 = conv2_block(c2, 256)
        clayer3 = conv1_block(flayer3, 256)
        at3 = attention_block(clayer3, fm_layer2, 16)
        T3 = Transpose_Block(flayer3, 128)
        c3 = concatenate([at3, T3], axis = 3)

        flayer2 = conv2_block(c3, 128)
        clayer2 = conv1_block(flayer2, 128)
        at4 = attention_block(clayer2, fm_layer1, 16)
        T4 = Transpose_Block(flayer2, 64)
        c4 = concatenate([at4, T4], axis = 3)

        
        flayer1 = conv2_block(c4, 64)

        final = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(flayer1)
        model = Model(input = inputs, output = final)
        dl=combined_loss()
        model.compile(optimizer = SGD(lr=5e-6, decay=1e-6, momentum=0.5, nesterov=True), loss = dl, metrics = ['accuracy', dice_score])
        if(pretrained_weights):
                model.load_weights(pretrained_weights)

        return model
