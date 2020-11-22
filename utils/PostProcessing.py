import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, unary_from_softmax, create_pairwise_gaussian
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
"""
#Original_image = Image which has to labelled
#Mask image = Which has been labelled by some technique..
def post_processing(data, probas):
    [h,w] = data.shape
    n_labels = 2
    pred_maps = np.zeros(data.shape)
    #print 'postprocessing:', data.shape, probas.shape
    img = np.uint8(data*255)
    img = data[...,np.newaxis]
    #proba = probas
    proba = probas + 0.1
    #proba[proba>1] = 1
    labels = np.zeros((2,img.shape[0],img.shape[1]))
    labels[0] = 1-proba
    labels[1] = proba
    #U=labels
    U = unary_from_softmax(labels)  # note: num classes is first dim
    pairwise_energy = create_pairwise_bilateral(sdims=(7,7), schan=(0.001,), img=img, chdim=2)
    pairwise_gaussian = create_pairwise_gaussian(sdims=(3,3), shape=img.shape[:2])

    d = dcrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_gaussian, compat=1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseEnergy(pairwise_energy, compat=1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)  # `compat` is the "strength" of this potential.

    Q = d.inference(5)
    pred_maps = np.argmax(Q, axis=0).reshape((h,w))
    return pred_maps
