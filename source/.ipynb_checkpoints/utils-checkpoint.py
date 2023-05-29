import numpy as np

def IoU(res, mask) :
    inter = np.logical_and(res, mask)
    union = np.logical_or(res, mask)
    
    iou_score = np.sum(inter) / np.sum(union)
    
    return iou_score

def postprocessing(res_seg) :
    
    res_seg[res_seg < 0.5] = 0
    res_seg[res_seg > 0.5] = 1

    where_0 = np.where(res_seg == 0)
    where_1 = np.where(res_seg == 1) 
 
    #res_seg[where_0] = 1
    #res_seg[where_1] = 0
    
    return(res_seg)

def merge_canal(img) :
    tmp = np.zeros((img.shape[1],img.shape[2]))
    for i in range(img.shape[0]) :
        tmp[img[i]==1] = i 
    return(tmp)