import numpy as np
from skimage import io 
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
import skimage.transform as trans
from Visualize import *
import scipy
import cv2


def hausdroff(mask, predict):
    return scipy.spatial.distance.directed_hausdorff(mask, predict)[0]/np.count_nonzero(mask)

def pixel_accuracy(mask, predict, targetsize = (256,256)):
    # mask , predict are numpy arrays which contains True to represent foreground
    # False to represent background
    total =  targetsize[0] * targetsize[1]
    z =np.count_nonzero(np.logical_xor(mask,predict))
    return (total-z)/total


def jaccard(mask, prediction, targetsize=(256,256)):
    P = np.count_nonzero(mask)
    N = np.count_nonzero(np.logical_not(mask))
    P_pred = np.count_nonzero(prediction)
    N_pred = np.count_nonzero(np.logical_not(prediction))
    TP = np.count_nonzero(np.logical_and(mask, prediction))
    TN = np.count_nonzero(np.logical_and(np.logical_not(mask), np.logical_not(prediction)))
    j_p = TP / (P + P_pred - TP)
    j_n = TN / (N + N_pred - TN)
    return (1/2)*(j_p + j_n)

def dice_coeff(mask, prediction, targetsize=(256,256)):
    TP = np.count_nonzero(np.logical_and(mask, prediction))
    FR = np.count_nonzero(np.logical_xor(mask, prediction))
    return (2*TP / (2*TP+FR))

def confusion_matrix(mask,prediction):
    Pos = np.count_nonzero(mask)
    Neg = np.count_nonzero(np.logical_not(mask))
    TP = np.count_nonzero(np.logical_and(mask, prediction))
    TN = np.count_nonzero(np.logical_not(np.logical_or(mask,prediction)))
    FP = np.count_nonzero(np.logical_and(mask,np.logical_not(prediction)))
    FN = np.count_nonzero(np.logical_and(np.logical_not(mask), prediction))
    return [TP,TN,FP,FN]

def quantitative_metrics(mask, prediction):
    [TP,TN,FP,FN] = confusion_matrix(mask,prediction)
    dice = dice_coeff(mask, prediction)
    jacc = jaccard(mask, prediction)
    OP = pixel_accuracy(mask, prediction)
    m = mIOU(mask, prediction)
    SEN = TP/(TP+FN)
    SPE = TN/(TN+FP)
    A = ((TP+FP)-(TP+FN))/ (TP+FN)
    A = np.absolute(A)
    h = hausdroff(mask, prediction)
    return [dice, jacc, OP, SEN, SPE, A, m, h]

def eval(mask_path, predicted_path):
    M = io.imread(mask_path) # Mask shape (4084,3328) , it has encoded fg=255 and bg=0
    P = io.imread(predicted_path) # (256,256)
    th = threshold_otsu(P)
    P[P>th] = 255
    P[P<=th] = 0
    M = trans.resize(M,(256,256))
    M = np.uint8(M *255)
    [dice, jacc, OP, SEN, SPE, A] = quantitative_metrics(M, P)
    print('Dice:', dice)
    print('jacc:', jacc)
    print('OP:', OP)
    print('SEN:', SEN)
    print('SPE:', SPE)
    print('A:',A)

def store_results(I, M, P, path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(256,256)
    mask_seg_map= draw_prediction_on_image(I, M, P)
    I = tfm.resize(I, (256,256))
    plt.axis('off')
    plt.imshow(I, cmap='gray')
    plt.imshow(mask_seg_map, alpha=0.5)
    plt.savefig(path,  bbox_inches='tight')

def store_results_path(I_path, mask_path, predicted_path, path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(256,256)
    M = io.imread(mask_path) # Mask shape (4084,3328) , it has encoded fg=255 and bg=0
    P = io.imread(predicted_path) # (256,256)
    I = io.imread(I_path)
    mask_seg_map= draw_prediction_on_image(I, M, P)
    '''
    I = tfm.resize(I, (256,256))
    plt.axis('off')
    plt.imshow(I, cmap='gray')
    plt.imshow(mask_seg_map, alpha=0.5)
    plt.show()
    #plt.savefig(path,  bbox_inches='tight')
    #plt.close(fig)
    '''
    
def mIOU(mask,prediction):
    [TP,TN,FP,FN] = confusion_matrix(mask,prediction)
    return (1/2)*((TP/(TP+FP+FN)) + (TN/(TN+FN+FP)))