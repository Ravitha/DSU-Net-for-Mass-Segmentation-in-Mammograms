import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.backend as K
from keras import optimizers
from keras import activations
from Metrics import *
from data import *
from keras_adamw import AdamW
def hybrid_forward(F,f1, f2):
        (N, H, W, C) = F.shape                          
        C = (int)(C/(f1*f2))                    
        x = Reshape( (f1*f2, H, W, C))(F)     # (N, f1*f2, H, W, C)
        x = Reshape(( f1*f2*H, W, C))(x)         # (N, f1*f2*H, W, C)
        x = Reshape( ( f1*H, f2, W, C))(x)      # (N, f1*H, f2, W, C)
        x = Reshape( ( f1*H, f2*W, C))(x)         # (N, f1*H, f2*W, C)
        return x    


def Basic(data, filter_size):
    #CONV=>BN=>RELU
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(data)
    conv_a = Activation('relu')(BatchNormalization()(conv))
    #CONV=>BN=>RELU
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(conv_a)
    conv = Activation('relu')(BatchNormalization()(conv))
    return conv

def resblock(data, filter_size):
    #CONV=>BN=>RELU
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(data)
    conv_a = Activation('relu')(BatchNormalization()(conv))
    #CONV=>BN=>RELU
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(conv_a)
    conv = Activation('relu')(BatchNormalization()(conv))
    #CONV=>BN
    conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(conv)
    conv_b = BatchNormalization()(conv)
    #RESNET CONNECTION =>RELU
    added = keras.layers.Add()([conv_b, conv_a])
    added = Activation('relu')(added)
    return added

def bilinear_Upsample(input_Tensor, channels):
     #UPSAMPLE
     buc = UpSampling2D(size = (2,2), interpolation='bilinear')(input_Tensor)
     #CONV=>REDUCE CHANNELS TO N
     buc = Conv2D(channels, 3, activation = 'relu', padding = 'same')(buc)
     return buc

def dense_Upsample(input_Tensor, channels):
    #CONV 4*CHANNELS => BN => RELU
    conv = Conv2D(4*channels, 3, activation = None, padding = 'same')(input_Tensor)
    conv = Activation('relu')(BatchNormalization()(conv))
    #PIXEL SHUFFLE
    duc = hybrid_forward(conv,2,2)
    return duc

def Channel_Attention(merge,channels):
    #GAP => Dense(2*c/16) => Dense(2*c) => multiply coefficients with Tensor
    gap = GlobalAveragePooling2D()(merge)
    fc1 = Dense((int)(2*channels/16), activation='relu')(gap)
    fc2 = Dense(2*channels, activation='sigmoid')(fc1)
    fc2 = Reshape((1,1,2*channels))(fc2)
    final = keras.layers.Multiply()([merge, fc2])
    return final

    
def concat(low_level, high_level, channels):
    #UPSAMPLE high_level
    buc = bilinear_Upsample(high_level,channels)
    duc = dense_Upsample(high_level,channels)
    #Add Low_level with duc(dense upsampling)
    added = keras.layers.Add()([low_level, duc])
    #CONV=>BN=>RELU
    added = Conv2D(channels, 3, activation = None, padding = 'same')(added)
    added = Activation('relu')(BatchNormalization()(added))
    #Concat added features with buc(bilinear upsampling)
    merge = concatenate([added,buc], axis = 3)
    #Channel Attention
    merge = Channel_Attention(merge,channels)
    return merge

def aunet(pretrained_weights = None, input_size = (256,256,1)):

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
    
    
    merge6 = concat(added4, added5, 512) 
    added6 = Basic(merge6, 512)
   
    
    merge7 = concat(added3, added6, 256)
    added7 = Basic(merge7, 256)

    merge8 = concat(added2, added7, 128)
    added8 = Basic(merge8, 128)

    merge9 = concat(added1, added8, 64)
    added9 = Basic(merge9, 64)
    
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(added9)
    model = Model(input = inputs, output = conv10)
    #dl = balanced_cross_entropy(0.8)
    dl=combined_loss()
    opt = AdamW(lr=1e-3, model = model, use_cosine_annealing=True, total_iterations = 24)
    #opt = keras.optimizers.Adam(lr=0.0001, amsgrad=True)
    model.compile(optimizer = opt, loss = dl, metrics = ['accuracy', dice_score])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

