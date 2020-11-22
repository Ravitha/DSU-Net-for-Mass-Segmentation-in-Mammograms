import tensorflow as tf
import keras
import numpy as np

def balanced_cross_entropy(beta):

  def convert_to_logits(y_pred):
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
      y_pred = convert_to_logits(y_pred)
      pos_weight = beta / (1 - beta)
      loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
      return tf.reduce_mean(loss * (1 - beta))

  return loss


def combined_loss():

  def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))
  
  def loss(y_true, y_pred, beta):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    return tf.reduce_mean(loss * (1 - beta))
    
  def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    numerator = 2 * tf.reduce_sum(tf.multiply(y_true,y_pred), axis=(1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    x = (numerator+1) / (denominator+1)
    final = tf.reduce_mean(x, axis=(0))
    return 1 - ((numerator+1) / (denominator+1))
  
  def combined(y_true, y_pred,):
    l1 =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return soft_dice_loss(y_true, y_pred) #+ l1
    #return 0.5 * soft_dice_loss(y_true, y_pred) + 0.5 *loss(y_true, y_pred, 0.8)
  return combined



def dice_score(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(tf.multiply(y_true,y_pred), axis=(1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    x = numerator / denominator
    final = tf.reduce_mean(x, axis=(0))
    return final
