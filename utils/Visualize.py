import numpy as np
from skimage import transform as tfm
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
def find_coords(mask, predict):
    seg_map = np.zeros((256,256,3), dtype = np.uint8) 
    #Color_Maps
    predict_color = (255,0,0)
    mask_color = (0,255,0)
    ind = 0
    for i in (mask, predict):
        coords = np.where(i == True)
        num = coords[0].shape[0]
        for pts in range(num):
          x = coords[0][pts]
          y = coords[1][pts]
          if(x!=0 and y!=0 and x!=255 and y!=255):
              if (not (i[x][y-1]==True and i[x][y+1]==True and i[x-1][y]==True and i[x+1][y]==True)==True):
                if ind==0:
                    seg_map[x,y,:] =mask_color
                else:
                    seg_map[x,y,:] = predict_color
        ind+=1
    return seg_map


def draw_prediction_on_image(I, mask, predict):
    #Resize mask and Image
    mask = tfm.resize(mask,(256,256))
    I = tfm.resize(I, (256,256))
    predict = np.reshape(predict,(256,256))
    #Threshold mask and predict
    thres = threshold_otsu(mask)
    mask = mask>=thres
    if(np.max(predict)!=np.min(predict)):
        thres_predict = threshold_otsu(predict)
    else:
        thres_predict = np.max(predict)
    predict = predict>=thres_predict
    print(mask.shape)
    print(predict.shape)
    #Find segmentation map
    seg_map = find_coords(mask, predict)
    plt.figure(figsize=(10,10))
    plt.imshow(I, cmap = 'gray')
    plt.imshow(seg_map, alpha=0.5)
    plt.show()
    return seg_map

