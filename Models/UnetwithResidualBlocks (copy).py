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
#UNET with Residual Blocks
def resblock(data, filter_size):
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(data)
    conv_a = Activation('relu')(BatchNormalization()(conv))
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_a)
    conv = Activation('relu')(BatchNormalization()(conv))
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)
    conv_b = BatchNormalization()(conv)
    added = keras.layers.Add()([conv_b, conv_a])
    added = Activation('relu')(added)
    return added

def unet_mini_three_resblock(pretrained_weights = None, input_size = (256,256,1)):

    inputs = Input(input_size)

    added1 = resblock(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(added1)

    added2 = resblock(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(added2)
    
    added3 = resblock(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(added3)
    

    added4 = resblock(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(added4)

    added5 = resblock(pool4, 1024)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2), interpolation='bilinear')(added5))
    merge6 = concatenate([added4,up6], axis = 3)
    
    added6 = resblock(merge6, 512)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2),interpolation='bilinear')(added6))
    merge7 = concatenate([added3,up7], axis = 3)
    
    added7 = resblock(merge7, 256)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2),interpolation='bilinear')(added7))
    merge8 = concatenate([added2,up8], axis = 3)

    added8 = resblock(merge8, 128)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2),interpolation='bilinear')(added8))
    merge9 = concatenate([added1,up9], axis = 3)

    added9 = resblock(merge9, 64)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(added9)
    model = keras.models.Model(input = inputs, output = conv10)
    dl = combined_loss()
    model.compile(optimizer = keras.optimizers.Adam(lr=0.001, amsgrad=True), loss = dl, metrics = ['accuracy', dice_score])
    #dl = balanced_cross_entropy(0.8)
    #model.compile(optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True), loss = dl, metrics = ['accuracy', dice_score])
    #model.compile(optimizer = optimizers.RMSprop(lr=0.0001), loss = dl, metrics = ['accuracy', dice_score])
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


    
