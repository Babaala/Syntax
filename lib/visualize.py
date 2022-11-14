import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy
import cv2
import random


#group a set of img patches 
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


def get_mode_index(target_arr):
    '''
    寻找众数，返回众数
    寻找第二多的数，返回该数
    '''
    vals, counts = np.unique(target_arr, return_counts=True)  # 返回一位数组唯一值，唯一值的个数

    the_mode_index = np.argmax(counts)  # 返回count最大值的索引。（寻找众数的位置）
    the_mode = vals[the_mode_index]  # 众数的值
    the_mode_num = counts[the_mode_index]  # 众数的个数

    max2 = np.sort(counts)[-2]  # 寻找第二多的数的个数
    max_index2 = np.argsort(counts)[-2]  # 寻找第二多的数的位置
    the_max2 = vals[max_index2]  # 寻找第二多的数的值

    return the_mode, the_mode_num, the_max2


def post_process(binary_img):
    binary_img = binary_img.astype(np.uint8).squeeze(-1)
    w, h = binary_img.shape
    img_color = np.zeros((w, h, 1), dtype=np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    # index = np.argmax(np.bincount(labels.flatten()))
    _, _, index = get_mode_index(labels.flatten())

    for x in range(w):
        for y in range(h):
            label = labels[x, y]
            if label != index:
                continue
            img_color[x, y, 0] = 1
            # img_color[x,y,:] = color[int(label)]
    return img_color

# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img,pred_res,gt, largest_connected):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))
    gt = data = np.transpose(gt,(1,2,0))

    binary = deepcopy(pred_res)

    if largest_connected:
        temp = deepcopy(pred_res)
        temp[temp>=0.1]=1
        temp[temp<0.1]=0
        post_pro = post_process(temp)

        binary[binary >= 0.5] = 1
        binary[binary < 0.5] = 0
        if ori_img.shape[2]==3:
            pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
            binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
            gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
            temp = np.repeat((temp*255).astype(np.uint8),repeats=3,axis=2)
            post_pro = np.repeat((post_pro*255).astype(np.uint8),repeats=3,axis=2)
        else:
            pred_res = (pred_res * 255).astype(np.uint8)
            binary = (binary * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)
            temp = (temp*255).astype(np.uint8)
            post_pro = (post_pro*255).astype(np.uint8)

        total_img = np.concatenate((ori_img,pred_res,binary,temp, post_pro,gt),axis=1)
        return total_img
    else:
        binary[binary >= 0.5] = 1
        binary[binary < 0.5] = 0
        if ori_img.shape[2] == 3:
            pred_res = np.repeat((pred_res * 255).astype(np.uint8), repeats=3, axis=2)
            binary = np.repeat((binary * 255).astype(np.uint8), repeats=3, axis=2)
            gt = np.repeat((gt * 255).astype(np.uint8), repeats=3, axis=2)
        else:
            pred_res = (pred_res * 255).astype(np.uint8)
            binary = (binary * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)
        total_img = np.concatenate((ori_img, pred_res, binary, gt), axis=1)
        return total_img

def see_concat_result(ori_img,pred_res, largest_connected):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))

    binary = deepcopy(pred_res)

    if largest_connected:
        temp = deepcopy(pred_res)
        temp[temp>=0.1]=1
        temp[temp<0.1]=0
        post_pro = post_process(temp)

        binary[binary >= 0.5] = 1
        binary[binary < 0.5] = 0
        if ori_img.shape[2]==3:
            pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
            binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
            temp = np.repeat((temp*255).astype(np.uint8),repeats=3,axis=2)
            post_pro = np.repeat((post_pro*255).astype(np.uint8),repeats=3,axis=2)
        else:
            pred_res = (pred_res * 255).astype(np.uint8)
            binary = (binary * 255).astype(np.uint8)
            temp = (temp*255).astype(np.uint8)
            post_pro = (post_pro*255).astype(np.uint8)

        total_img = np.concatenate((ori_img,pred_res,binary,temp, post_pro),axis=1)
        return total_img
    else:
        binary[binary >= 0.5] = 1
        binary[binary < 0.5] = 0
        if ori_img.shape[2] == 3:
            pred_res = np.repeat((pred_res * 255).astype(np.uint8), repeats=3, axis=2)
            binary = np.repeat((binary * 255).astype(np.uint8), repeats=3, axis=2)
        else:
            pred_res = (pred_res * 255).astype(np.uint8)
            binary = (binary * 255).astype(np.uint8)
        total_img = np.concatenate((ori_img, pred_res, binary), axis=1)
        return total_img


#visualize image, save as PIL image
def save_img(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  #the image is between 0-1
    img.save(filename)
    return img