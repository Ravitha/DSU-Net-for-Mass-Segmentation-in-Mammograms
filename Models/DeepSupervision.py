import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras import regularizers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as keras
import keras.backend as K
from keras import optimizers
from keras import activations
from Metrics import *
import tensorflow as tf
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from keras.regularizers import l1_l2
from keras_adamw import AdamW


class GetCurrentEpoch(Callback):
    def __init__(self, current_epoch):
        super(GetCurrentEpoch, self).__init__()
        self.current_epoch = current_epoch

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.current_epoch, epoch)

t = K.variable(1.0)
T = K.constant(20.0)
def composite_loss(layer1, layer2, layer3, layer4, layer5, layer6, fusion):
  
    def be(x):
        ori = np.copy(x)
        mod = np.copy(x)
        for i in range(x.shape[0]):
            img = mod[i]
            img = img.reshape((256,256))
            eroded = binary_erosion(img,disk(3))
            mod[i] = eroded.reshape((256,256,1))
            mod[i] = ori[i] - mod[i]
        return mod
    
    def cross_entropy(y_true, y_pred, epsilon = 0.8):
        y_pred = y_pred + epsilon
        posprob = -K.log(y_pred)
        negprob = -K.log(1-y_pred)
        prob = 0.7 * tf.multiply(y_true, posprob) + 0.3 * tf.multiply((1-y_true), negprob)
        return tf.reduce_mean(tf.reduce_mean(prob, axis=(1,2)))

    def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
        numerator = 2 * tf.reduce_sum(tf.multiply(y_true,y_pred), axis=(1,2))
        denominator = tf.reduce_sum(K.square(y_true) +K.square(y_pred), axis=(1,2))
        x = (numerator+1) / (denominator+1)
        x = tf.reduce_mean(x)
        return  (-K.log(x))**0.3#0.3
    
    def combined(y_true, y_pred):
        eroded = tf.compat.v1.placeholder(tf.float32,(None,256,256,1))   
        erode = tf.numpy_function(be,[y_true],[tf.float32])
        eroded = erode[0]
        eroded = tf.ensure_shape(eroded, (None, 256, 256,1))
        return soft_dice_loss(y_true, y_pred) + (1-t/100)**2 *  (soft_dice_loss(eroded, layer1) \
             +  soft_dice_loss(eroded, layer2) \
             +  soft_dice_loss(eroded, layer3) + \
                soft_dice_loss(y_true, layer4) + \
                soft_dice_loss(y_true, layer5) + \
                soft_dice_loss(y_true, layer6) + \
                soft_dice_loss(y_true, fusion)) 
    
    return combined



def Channel_Attention(merge,channels):
    #GAP => Dense(c/16) => Dense(c) => multiply coefficients with Tensor
    gap = GlobalAveragePooling2D()(merge)
    fc1 = Dense((int)(channels/16), activation='relu')(gap)
    fc2 = Dense(channels, activation='sigmoid')(fc1)
    fc2 = Reshape((1,1,channels))(fc2)
    final = keras.layers.Multiply()([merge, fc2])
    return final

def encoder(data, filter_size):
     conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(data)
     conv_a = Activation('relu')(BatchNormalization()(conv))
     conv = Conv2D(filter_size, 3, activation = None, padding = 'same')(conv_a)
     conv = Activation('relu')(BatchNormalization()(conv))
     return conv
 
def attentionblock(data, size):
     buc = Conv2D(1, 1, activation = None, padding = 'same')(data)
     buc = Activation('relu')(BatchNormalization()(buc))
     buc = UpSampling2D(size = size, interpolation='bilinear')(buc)
     return buc 


def decoder(lfeatures, hfeatures, channels):
    lfeatures = Channel_Attention(lfeatures, channels)
    hfeatures = Channel_Attention(hfeatures, 2*channels)
    hfeatures = (UpSampling2D(size = (2,2), interpolation='bilinear')(hfeatures))
    conv = Conv2D(channels, 3, activation = None, padding = 'same')(hfeatures)
    conv_a = Activation('relu')(BatchNormalization()(conv))
    conv = Add()([lfeatures, conv_a])
    conv = Conv2D(channels, 3, activation = None, padding = 'same')(conv)
    conv_a = Activation('relu')(BatchNormalization()(conv))
    conv = Conv2D(channels, 3, activation = None, padding = 'same')(conv_a)
    conv = Activation('relu')(BatchNormalization()(conv))
    return conv

def deepSupervision(pretrained_weights = None, input_size = (256,256,3)):

    inputs = Input(input_size)
    #Encoder 1
    added1 = encoder(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(added1)
    
    #Encoder 2
    added2 = encoder(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(added2)
    att1 = attentionblock(added2,(2,2))

    #Encoder 3
    added3 = encoder(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(added3)
    att2 = attentionblock(added3,(4,4))
    
    #Encoder 4
    added4 = encoder(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(added4)
    att3 = attentionblock(added4,(8,8))

    #Encoder 5
    added5 = encoder(pool4, 1024)

    #Decoder 1
    merge6 = decoder(added4,added5, 512)
    added6 = Dropout(0.5)(merge6)
    #added6 = merge6
    att4 = attentionblock(added6,(8,8))
    
    #Decoder 2
    merge7 = decoder(added3,added6, 256)
    added7 = Dropout(0.5)(merge7)
    #added7 = merge7
    att5 = attentionblock(added7,(4,4))

    #Decoder 3
    merge8 = decoder(added2,added7, 128)
    added8 = Dropout(0.5)(merge8)
    #added8 = merge8
    att6 = attentionblock(added8,(2,2))

    #Decoder 4
    added9 = decoder(added1,added8, 64)
    added9 = Dropout(0.5)(added9)
   
    
    added9 = concatenate([added9,att1,att2,att3,att4,att5,att6], axis=3)
    added10 = Conv2D(1, 1, activation = 'sigmoid', padding = 'same')(added9)
    
    model = Model(input = inputs, output = added10)
    dl = composite_loss(att1,att2,att3,att4,att5,att6,added9)
    opt = AdamW(lr=1e-3, model = model, use_cosine_annealing=True, total_iterations = 24)
    model.compile(optimizer = opt, loss = dl, metrics = ['accuracy', dice_score])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
