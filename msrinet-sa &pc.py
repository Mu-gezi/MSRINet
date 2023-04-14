#!/usr/bin/env python
# coding: utf-8


# 引入所需的包
from itertools import count
from unicodedata import name
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.utils.data import Dataset, DataLoader


# import keras
# from keras.utils import np_utils # np_utils.to_categorical将类别向量映射为二值类别矩阵

from sklearn.decomposition import PCA # PCA降维
from sklearn.model_selection import train_test_split # 训练集和验证集分类
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score # 混淆矩阵，精确度，分类报告和卡帕系数
from sklearn.feature_extraction import image
from operator import truediv # operator.truediv(a, b)返回 a/ b

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio # 用于读取.mat文件
# import skimage.io
import os # 用于获取路径
import h5py
import spectral


# # 1. Define Global Variables
# GLOBAL VARIABLES
dataset = 'SA'
Bands = 30
samples = 20
train_ratio = 0.1
windowSize = 15  # 设置邻域窗口大小
smallwindowSize = 3
ca = 16
randomState = 3456# 3456
# modelpath = os.path.join('model-PU', 'best' + '10' + '-' + 'HU' + '.pkl')
# resultpath = os.path.join('model-PU', 'classification_report' + '5' + '-' + 'HU18' + '20samples' + '.txt')

modelpath = os.path.join('model-SA', 'best' + '-' + '20samples' + '.pkl')
resultpath = os.path.join('model-SA', 'classification_report' + '-' + '20samples' + '.txt')
imagepath = os.path.join('model-SA', 'predictions-6.8' + '.jpg')


# 按命名导入数据
def loadData(name):
    if name == 'PU':
        data = sio.loadmat('高光谱数据集/高光谱数据集/Pavia/paviaU.mat')['paviaU']
        labels = sio.loadmat('高光谱数据集/高光谱数据集/Pavia/paviaU_gt.mat')['Data_gt']
    # elif name == 'H18':
    #     data = hdf5storage.loadmat('Houston/Houston_18.mat')['houstonU']
    #     labels = hdf5storage.loadmat('Houston/Houston_18_gt.mat')['houstonU_gt']
    elif name == 'IP':
        data = sio.loadmat('高光谱数据集/高光谱数据集/Idian_pines/indian_pines.mat')['HSI_original']
        labels = sio.loadmat('高光谱数据集/高光谱数据集/Idian_pines/indian_pines_gt.mat')['Data_gt']
    elif name == 'SA':
        data = sio.loadmat('高光谱数据集/高光谱数据集/Salinas/salinas.mat')['HSI_original']
        labels = sio.loadmat('高光谱数据集/高光谱数据集/Salinas/salinas_gt.mat')['Data_gt']
    elif name == 'PC':
        data = sio.loadmat('高光谱数据集/高光谱数据集/Pavia/pavia.mat')['HSI_original']
        labels = sio.loadmat('高光谱数据集/高光谱数据集/Pavia/pavia_gt.mat')['Data_gt']
    elif name == 'HU18':
        data = np.array(h5py.File('高光谱数据集/高光谱数据集/HoustonU/Houston_18.mat')['houstonU'])
        data = data.T
        labels = np.array(h5py.File('高光谱数据集/高光谱数据集/HoustonU/Houston_18_gt.mat')['houstonU_gt'])
        labels = labels.T
    
    return data, labels


# # 2. Spatial Channel Data Extraction
# 定义数据划分函数
def splitTrainTestSet(X, y, trainRatio, randomState=randomState):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=trainRatio,
        random_state=randomState)# ,stratify=y) # 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
    return X_train, X_test, y_train, y_test


def splitTrainTestSet_Num1(X, y, Num, category, randomState=345):
    import random
    cate = np.arange(category)
    for i in cate:
        index = np.argwhere(y == i)
        np.random.seed(randomState)
        np.random.shuffle(index)
        if i == 0:
            index_train = index[0:Num]
            index_test = index[Num:]
        elif i == 7:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
        elif i == 14:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
        else:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
    index_train = index_train.flatten().tolist()
    index_test = index_test.flatten().tolist()
    X_train = X[index_train,:]
    X_test = X[index_test,:]
    y_train = y[index_train]
    y_test = y[index_test]
    return X_train, X_test, y_train, y_test


# 定义PCA降维函数
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2])) # 这里的-1被理解为unspecified value，意思是未指定为给定的。
    # 使用“-1”是因为python可以自动根据后面的值和剩余维数推出前面的值
    # numpy的三维数组中，“.shape”的返回值为三个数的元组，分别表示页、行和列，且axis=0表示在页的方向上操作，axis=1表示在列的方向上来操作，axis=2表示在行的方向来操作
    # 若以IP数据为例，则读入的数据是145行200列145页的数据，那么在此处reshape的时候，相当于保持光谱维度不变，按页进行平铺
    # 需要注意reshape新生成数组和原数组公用一个内存，不管改变哪个都会互相影响。
    pca = PCA(n_components=numComponents, whiten=True) #   whiten即为白化，目的是每个特征具有相同的方差
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents)) # 将降维后的数据还原
    var_ratio = pca.explained_variance_ratio_
#     for idx, val in enumerate(var_ratio, 1):
#         print("Principle component%d:%.2f%%" % (idx, val * 100))
#         print("total:%.2f%%" % np.sum(var_ratio * 100))
        
    return newX, pca


# 添加宽度为2的补丁
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2])) # 设定宽度为2的边框补丁
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X # 将中间0区域用原始数据填满
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin) # 生成添加pad的数据
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])) # 初始化patchsData
    patchesLabels = np.zeros((X.shape[0] * X.shape[1])) # 初始化patchsLabels
    patchIndex = 0
    # 通过二重循环提取一系列patchs
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels: # 通过标签大于0来过滤掉无标签的数据
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1 # 为方便后面通过下标查找类别时可以对应上
    return patchesData, patchesLabels


def extractInformativePixel(X, y, removeZeroLabels = True):
    # split pixels
    pixelsData = np.zeros((X.shape[0] * X.shape[1], X.shape[2])) # 初始化patchsData
    pixelsLabels = np.zeros((X.shape[0] * X.shape[1])) # 初始化patchsLabels
    pixelIndex = 0
    # 通过二重循环提取一系列pixels
    for r in range(0, X.shape[0]):
        for c in range(0, X.shape[1]):
            pixel = X[r, c]   
            pixelsData[pixelIndex, :] = pixel
            pixelsLabels[pixelIndex] = y[r, c]
            pixelIndex = pixelIndex + 1
    if removeZeroLabels: # 通过标签大于0来过滤掉无标签的数据
        pixelsData = pixelsData[pixelsLabels>0,:]
        pixelsLabels = pixelsLabels[pixelsLabels>0]
        pixelsLabels -= 1 # 为方便后面通过下标查找类别时可以对应上
    return pixelsData, pixelsLabels


# ## 1.1 Data Loading
# 导入数据
X, y = loadData(dataset)
# 观察数据维度信息
X.shape, y.shape


# 保存原始维度
original_Bands = X.shape[2]
print(original_Bands)

# 使用pca进行降维
X,pca = applyPCA(X,numComponents=Bands)
print(X.shape)


# ## 1.2 Neighbourhood Extraction
# Neighbourhood Extraction（邻域提取），即将原始数据立方体重复提取S*S*B的小立方体（其中S是windowSize，B是spectral band的数量），用中心label来代表一个立方体
X, y = createImageCubes(X, y, windowSize=windowSize)

print(X.shape, y.shape)


# ## 1.3 Center Variables Extraction
# 初始化中心向量
X_center = np.zeros((X.shape[0],1,1,Bands))

for i in range(X.shape[0]):
    X_center[i:,:,:] = X[i,int((windowSize-1)/2),int((windowSize-1)/2),:]
X_center = np.squeeze(X_center, axis=(1,2))
y_center = y
print(X_center.shape)
print(y_center.shape)


# ## 1.4 训练集与测试集划分
# X_train, X_test, y_train, y_test = splitTrainTestSet_Num(X, y, 120, 9, randomState=345)
from collections import Counter

X_train, X_test, y_train, y_test = splitTrainTestSet_Num1(X, y, samples, ca, randomState=randomState)
print('Counter(data)\n',Counter(y_train))
print('Counter(data)\n',Counter(y_test))
# y_train = np_utils.to_categorical(y_train) # 将类别向量映射为二值类别矩阵
# y_test = np_utils.to_categorical(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('/n')


X_center_train, X_center_test, y_center_train, y_center_test = splitTrainTestSet_Num1(
    X_center, y_center, samples, ca, randomState=randomState)
# y_center_train = np_utils.to_categorical(y_center_train) # 将类别向量映射为二值类别矩阵
# y_center_test = np_utils.to_categorical(y_center_test)
print(X_center_train.shape, X_center_test.shape, y_center_train.shape, y_center_test.shape)


# ## 1.5 Data Augmentation
# # 数据增广
# X90 = np.zeros((X_train.shape[0], windowSize, windowSize, Bands), dtype = 'int32')
# X180 = np.zeros((X_train.shape[0], windowSize, windowSize, Bands), dtype = 'int32')
# X270 = np.zeros((X_train.shape[0], windowSize, windowSize, Bands), dtype = 'int32')

# for i in range(X_train.shape[0]):
#     X90_temp = np.rot90(X[i,:,:,:].swapaxes(0, 1)).swapaxes(0, 1)
#     X90[i,:,:,:] = X90_temp

# for i in range(X_train.shape[0]):
#     X180_temp = np.rot90(X[i,:,:,:].swapaxes(0, 1), 2).swapaxes(0, 1)
#     X180[i,:,:,:] = X180_temp
    
# for i in range(X_train.shape[0]):
#     X270_temp = np.rot90(X[i,:,:,:].swapaxes(0, 1), 3).swapaxes(0, 1)
#     X270[i,:,:,:] = X270_temp
    
# X_aug_train = np.concatenate((X_train, X90, X180, X270), axis=0)
# y_aug_train = np.concatenate((y_train,y_train,y_train,y_train), axis=0)
# print(X_aug_train.shape)
# print(y_aug_train.shape)


# X_aug_center_train = np.concatenate((X_center_train, X_center_train, X_center_train, X_center_train), axis=0)
# y_aug_center_train = np.concatenate((y_center_train,y_center_train,y_center_train,y_center_train), axis=0)
# print(X_aug_center_train.shape)
# print(y_aug_center_train.shape)


# # 2. Spectral Channel Data Extraction
# 创建小窗口patchs
def createSmallImageCubes(X, y, batch, windowSize=5):# , removeZeroLabels = True
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin) # 生成添加pad的数据
    # split patches
    patchesData = np.zeros((batch, windowSize, windowSize, X.shape[2])) # 初始化patchsData
    patchesLabels = np.zeros((batch)) # 初始化patchsLabels
    patchIndex = 0
    # 通过二重循环提取一系列patchs
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            Label = y[r-margin, c-margin]
            if Label == 0:
                continue
            else:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
#     if removeZeroLabels: # 通过标签大于0来过滤掉无标签的数据
#         patchesData = patchesData[patchesLabels>0,:,:,:]
#         patchesLabels = patchesLabels[patchesLabels>0]
    patchesLabels -= 1 # 为方便后面通过下标查找类别时可以对应上
    return patchesData, patchesLabels


# 提取有信息的像素
def extractInformativePixel(X, y, removeZeroLabels = True):
    # split pixels
    pixelsData = np.zeros((X.shape[0] * X.shape[1], X.shape[2])) # 初始化patchsData
    pixelsLabels = np.zeros((X.shape[0] * X.shape[1])) # 初始化patchsLabels
    pixelIndex = 0
    # 通过二重循环提取一系列pixels
    for r in range(0, X.shape[0]):
        for c in range(0, X.shape[1]):
            pixel = X[r, c]   
            pixelsData[pixelIndex, :] = pixel
            pixelsLabels[pixelIndex] = y[r, c]
            pixelIndex = pixelIndex + 1
    if removeZeroLabels: # 通过标签大于0来过滤掉无标签的数据
        pixelsData = pixelsData[pixelsLabels>0,:]
        pixelsLabels = pixelsLabels[pixelsLabels>0]
        pixelsLabels -= 1 # 为方便后面通过下标查找类别时可以对应上
    return pixelsData, pixelsLabels


# 提取有信息的像素下标
def extractInformativePixelIndex(X, y, removeZeroLabels = True):
    # split pixels
    pixelsData = np.zeros((X.shape[0] * X.shape[1], X.shape[2])) # 初始化patchsData
    pixelsLabels = np.zeros((X.shape[0] * X.shape[1])) # 初始化patchsLabels
    pixelIndex = 0
    # 通过二重循环提取一系列pixels
    for r in range(0, X.shape[0]):
        for c in range(0, X.shape[1]):
            pixel = X[r, c]   
            pixelsData[pixelIndex, :] = pixel
            pixelsLabels[pixelIndex] = y[r, c]
            pixelIndex = pixelIndex + 1
    if removeZeroLabels: # 通过标签大于0来过滤掉无标签的数据
        pixelsData = pixelsData[pixelsLabels>0,:]
        pixelsLabels = pixelsLabels[pixelsLabels>0]
        pixelsLabels -= 1 # 为方便后面通过下标查找类别时可以对应上
    return pixelsData, pixelsLabels


# 提取固定个数训练数据
def splitTrainTestSet_Num2(X, y, Num, category, randomState=randomState):
    import random
    cate = np.arange(category)
    for i in cate:
        index = np.argwhere(y == i)
        np.random.seed(randomState)
        np.random.shuffle(index)
        if i == 0:
            index_train = index[0:Num]
            index_test = index[Num:]
        elif i == 7:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
        elif i == 14:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
        else:
            index_train = np.concatenate([index_train, index[0:Num]])
            index_test = np.concatenate([index_test, index[Num:]])
    index_train = index_train.flatten().tolist()
    index_test = index_test.flatten().tolist()
    X_train = X[index_train,:]
    X_test = X[index_test,:]
    y_train = y[index_train]
    y_test = y[index_test]
    return X_train, X_test, y_train, y_test, index_train, index_test


# ## 2.1 Data Loading
# 导入数据
X, y = loadData(dataset)
# 观察数据维度信息
X.shape, y.shape


# 提取有信息的像素向量
X_pixel_initial, y_pixel_initial  = extractInformativePixel(X, y)

print(X_pixel_initial.shape, y_pixel_initial.shape)


# 获取训练集和测试集下标
from collections import Counter

X_pixel_initial_train, X_pixel_initial_test, y_pixel_initial_train, y_pixel_initial_test, index_train, index_test = splitTrainTestSet_Num2(
    X_pixel_initial, y_pixel_initial, samples, ca, randomState=randomState)
# del a, b
# print('Counter(data)\n',Counter(y_pixel_initial_train))
# print('Counter(data)\n',Counter(y_pixel_initial_test))
index_train.sort()
index_test.sort()
print(np.unique(y_pixel_initial_train))
print(len(index_train))
print(len(index_test))


# ## 2.2 Obtain Training and Testing Matrix from Original Image
# 训练集矩阵
y_flatten = y.reshape(1,-1)

count_non_zero = 0
count_index = 0
count_zero = 0
for i in np.arange(y_flatten.shape[1]):
    if y_flatten[0,i] == 0:
        count_zero += 1
    elif y_flatten[0,i] > 0:
        count_non_zero += 1
        
    if count_non_zero-1 == index_train[count_index]:
        y_flatten[0,count_zero+count_non_zero-1] = y_flatten[0,count_zero+count_non_zero-1]+100
        count_index += 1
    
    if count_index == len(index_train):
        break
        
print(np.unique(y_flatten))
for j in np.arange(y_flatten.shape[1]):
    if y_flatten[0,j] > ca:
        y_flatten[0,j] = y_flatten[0,j]-100
    else:
        y_flatten[0,j] = 0

y_train_matrix = y_flatten.reshape(y.shape[0], y.shape[1])
np.unique(y_train_matrix)
print(y_train_matrix.shape)
print(np.count_nonzero(y_train_matrix)) 


# 测试集矩阵
y_flatten = y.reshape(1,-1)

count_non_zero = 0
count_index = 0
count_zero = 0
for i in np.arange(y_flatten.shape[1]):
    if y_flatten[0,i] == 0:
        count_zero += 1
    elif y_flatten[0,i] > 0:
        count_non_zero += 1
        
    if count_non_zero-1 == index_test[count_index]:
        y_flatten[0,count_zero+count_non_zero-1] = y_flatten[0,count_zero+count_non_zero-1]+100
        count_index += 1
    
    if count_index == len(index_test):
        break
        
print(np.unique(y_flatten))
for j in np.arange(y_flatten.shape[1]):
    if y_flatten[0,j] > ca:
        y_flatten[0,j] = y_flatten[0,j]-100
    else:
        y_flatten[0,j] = 0

y_test_matrix = y_flatten.reshape(y.shape[0], y.shape[1])
np.unique(y_test_matrix)
print(y_test_matrix.shape)
print(np.count_nonzero(y_test_matrix)) 


# ## 2.3 Neighbourhood Extraction
# Neighbourhood Extraction（邻域提取），即将原始数据立方体重复提取S*S*B的小立方体（其中S是windowSize，B是spectral band的数量），用中心label来代表一个立方体
# 训练集
X_small_train, y_small_train = createSmallImageCubes(X, y_train_matrix, np.count_nonzero(y_train_matrix), windowSize=3)
print(X_small_train.shape, y_small_train.shape)

X_small_test, y_small_test = createSmallImageCubes(X, y_test_matrix, np.count_nonzero(y_test_matrix), windowSize=3)
print(X_small_test.shape, y_small_test.shape)


# ## 2.4 Data Augmentation
# # 数据增广
# X90 = np.zeros((X_small_train.shape[0], smallwindowSize, smallwindowSize, original_Bands), dtype = 'int32')
# X180 = np.zeros((X_small_train.shape[0], smallwindowSize, smallwindowSize, original_Bands), dtype = 'int32')
# X270 = np.zeros((X_small_train.shape[0], smallwindowSize, smallwindowSize, original_Bands), dtype = 'int32')

# for i in range(X_small_train.shape[0]):
#     X90_temp = np.rot90(X_small_train[i,:,:,:].swapaxes(0, 1)).swapaxes(0, 1)
#     X90[i,:,:,:] = X90_temp

# for i in range(X_small_train.shape[0]):
#     X180_temp = np.rot90(X_small_train[i,:,:,:].swapaxes(0, 1), 2).swapaxes(0, 1)
#     X180[i,:,:,:] = X180_temp
    
# for i in range(X_small_train.shape[0]):
#     X270_temp = np.rot90(X_small_train[i,:,:,:].swapaxes(0, 1), 3).swapaxes(0, 1)
#     X270[i,:,:,:] = X270_temp
    
# X_aug_small_train = np.concatenate((X_small_train, X90, X180, X270), axis=0)
# y_aug_small_train = np.concatenate((y_small_train,y_small_train,y_small_train,y_small_train), axis=0)
# print(X_aug_small_train.shape)
# print(y_aug_small_train.shape)


# ## 2.5 Generate SBNL Data
# initial_matrix_odd = np.zeros((
#     X_small_test.shape[0],
#     X_small_test.shape[1],
#     X_small_test.shape[2],
#     X_small_test.shape[3]*2))
# initial_matrix_even = np.zeros((
#     X_small_test.shape[0],
#     X_small_test.shape[1],
#     X_small_test.shape[2],
#     X_small_test.shape[3]*2))
# for i in range(X_small_test.shape[3]*2):
#     j = 0
#     if (i % 2) == 0:
#         initial_matrix_even[:,:,:,i] = X_small_test[:,:,:,j]
#         initial_matrix_odd[:,:,:,i+1] = X_small_test[:,:,:,X_small_test.shape[3]-1-i-1]
        
# SBNL_test = initial_matrix_even+initial_matrix_odd
# print(SBNL_test.shape)


# # In[30]:


# initial_matrix_odd = np.zeros((
#     X_aug_small_train.shape[0],
#     X_aug_small_train.shape[1],
#     X_aug_small_train.shape[2],
#     X_aug_small_train.shape[3]*2))
# initial_matrix_even = np.zeros((
#     X_aug_small_train.shape[0],
#     X_aug_small_train.shape[1],
#     X_aug_small_train.shape[2],
#     X_aug_small_train.shape[3]*2))
# for i in range(X_aug_small_train.shape[3]*2):
#     j = 0
#     if (i % 2) == 0:
#         initial_matrix_even[:,:,:,i] = X_aug_small_train[:,:,:,j]
#         initial_matrix_odd[:,:,:,i+1] = X_aug_small_train[:,:,:,X_aug_small_train.shape[3]-1-i-1]
        
# SBNL = initial_matrix_even+initial_matrix_odd
# print(SBNL.shape)

# # 3. Model and Training
class involution(nn.Module):
    def __init__(self,
                channels,
                kernel_size,
                stride,
                reduct=2,
                groups=2):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = reduct
        self.groups = groups
        self.group_channels = self.channels // self.groups
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class IVoubottleneck(nn.Module):
    def __init__(self, inplanes, planes, iksize=[3,5,7], reduct=2, groups=2):
        super(IVoubottleneck, self).__init__()
        self.conv1 = involution(inplanes, iksize[0], 1, reduct, groups)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(True)

        self.conv2 = involution(inplanes, iksize[1], 1, reduct, groups)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.conv3 = involution(inplanes, iksize[2], 1, reduct, groups)
        self.bn3 = nn.BatchNorm2d(inplanes)

        self.conv = nn.Conv2d(planes, inplanes, 1, 1, 0, bias=False)
        
    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        out = torch.cat([out1,out2,out3], dim=1)
        out = self.conv(out)

        out += identity

        return out
    
    
class GlobalCovPooling(nn.Module):
    def __init__(self, batch, h, w, c, d):
        super(GlobalCovPooling, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.conv2d1 = nn.Conv2d(c, d, 1, 1, 0, bias=False)
        self.conv2d2 = nn.Conv2d(d, c, 1, 1, 0, bias=False)
    
    def forward(self, x):
        x = x.permute(0,2,3,1)
        b,h,w,c = x.shape
        x = torch.reshape(x, (x.size(0), x.size(1)*x.size(2), x.size(3)))
        x_T = x.permute(0,2,1)
        sigma = torch.bmm(x_T,(1/x.size(1)*(torch.eye(x.size(1), x.size(1)).unsqueeze(dim=2).detach().repeat(1,1,x.size(0))).permute(2,0,1) - 1/(x.size(1)*x.size(1))*torch.ones(x.size(0), x.size(1), x.size(1))).to(device))
        sigma = torch.bmm(sigma,x)
        
        temp = torch.tensor(np.arange(sigma.size(0)))
        for j in range(sigma.size(0)):
            temp[j] = 1/torch.trace(sigma[j,:,:])
        temp = torch.reshape(temp,(sigma.size(0),1,1)).to(device)
        sigma = temp*sigma.to(device)
        
        Y = sigma
        Z = (torch.eye(sigma.size(1),sigma.size(2)).unsqueeze(dim=2).detach().repeat(1,1,x.size(0))).permute(2,0,1).to(device)
        for i in range(6):
            Y_temp = Y.to(device)
            Y = (1/2*Y).to(device)*(3*(torch.eye(sigma.size(1),sigma.size(2)).unsqueeze(dim=2).detach().repeat(1,1,x.size(0))).permute(2,0,1).to(device)-torch.bmm(Z,Y)).to(device)
            Z = torch.bmm(1/2*(3*(torch.eye(sigma.size(1),sigma.size(2)).unsqueeze(dim=2).detach().repeat(1,1,x.size(0))).permute(2,0,1).to(device)-torch.bmm(Z,Y_temp)).to(device), Z)
        
        temp = torch.tensor(np.arange(Y.size(0)))
        for j in range(Y.size(0)):
            temp[j] = torch.trace(Y[j,:,:]).sqrt()
        temp = torch.reshape(temp,(Y.size(0),1,1)).to(device)
        Y_hat = temp*Y.to(device)
        Y_hat = Y_hat.mean(1)
        Y_hat = torch.reshape(Y_hat,(Y_hat.size(0), -1, 1, 1))
#         Y_hat = self.relu(self.conv2d2(self.sigmoid(self.conv2d1(Y_hat))))
        return Y_hat
    
    
class lip2d(nn.Module):
    def __init__(self, in_channels, kernel, stride, padding):
        super(lip2d, self).__init__()
        self.logit_module = nn.Conv2d(in_channels, in_channels, 1)
        self.avgpool2d = nn.AvgPool2d(kernel, stride, padding)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.logit_module(x)
        weight = self.sigmoid(out)
        out = self.avgpool(x*weight)# self.avgpool(weight)
        return out
        
    
class cross_attention(nn.Module):
    def __init__(self, inplanes, planes, iksize=3, reduct=2, groups=[1,1,204]):
        super(cross_attention, self).__init__()
        self.conv1_local = involution(inplanes, iksize, 1, reduct, groups[2])
        self.conv2_global = involution(inplanes, iksize, 1, reduct, groups[0])
        self.conv3 = involution(inplanes, iksize, 1, reduct, groups[1])
        self.conv4 = involution(102, iksize, 1, reduct, groups[1])
        self.conv5 = involution(51, iksize, 1, reduct, groups[1])
        
        self.conv2d1 = nn.Conv2d(inplanes, 102, 1, 1, 0, bias=False)
        self.conv2d2 = nn.Conv2d(102, 51, 1, 1, 0, bias=False)
        self.conv2d3 = nn.Conv2d(51, 16, 1, 1, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(True)

        self.bn2 = nn.BatchNorm2d(102)

        self.bn3 = nn.BatchNorm2d(51)

    def forward(self, x):
        out_local = self.conv1_local(x)
        out_global = self.conv2_global(x)
        product = out_local+out_global
        output = x*product
        
        identity = output
        out = self.conv3(output)
        out = self.bn1(out)
        out = self.relu(out)
        out = out+identity
        out = self.conv2d1(out)

        identity = out
        out = self.conv4(out)
        out = self.bn2(out)
        out = self.relu(out)
#         out = out+identity
        out = self.conv2d2(out)

        identity = out
        out = self.conv5(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = out+identity
        out = self.conv2d3(out)
        return out


class myNet(nn.Module):
    def __init__(self, num_classes, channels, args):
        super(myNet, self).__init__()
        self.reduct = args.num_involution_reduct
        self.groups = args.num_involution_groups
        self.iksize = args.involution_kernel_size
        self.numblocks = args.num_blocks
        self.batch = args.num_batch
        self.patchheight = args.patch_height
        self.patchwidth = args.patch_width
        self.patchchannels = args.num_patch_channels
        self.inplanes = 30
        self.inplanes_crossattention = original_Bands
        self.midinplanes = 90
        
        layers = []
        for _ in range(self.numblocks):
            layers.append(IVoubottleneck(self.inplanes, self.midinplanes, iksize=self.iksize, reduct=self.reduct, groups=self.groups))
        self.block = nn.Sequential(*layers)

        self.bn = nn.BatchNorm2d(46)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpoolone = nn.AdaptiveAvgPool2d((10,10))
        self.avgpooltwo = nn.AdaptiveAvgPool2d((5,5))
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.maxpoolone = nn.AdaptiveMaxPool2d((10,10))
        self.maxpooltwo = nn.AdaptiveMaxPool2d((5,5))
        self.covpool1 = GlobalCovPooling(self.batch, self.patchheight, self.patchwidth, self.patchchannels, 15)
        self.covpool2 = GlobalCovPooling(self.batch, self.patchheight, self.patchwidth, 16, 8)
        self.lip1 = lip2d(30,15,1,0)
        self.lip2 = lip2d(16,3,1,0)

        self.conv2d1 = nn.Conv2d(60, 30, 1, 1, 0, bias=False)
        self.conv2d2 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)
        self.inv = involution(self.inplanes, 3, 1, 2, 2)
        
        self.linear = nn.Linear(46, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.cross_attention = cross_attention(self.inplanes_crossattention, self.midinplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        out = self.block(x)
#         out = self.avgpoolone(out)+self.maxpoolone(out)
#         out = self.relu(self.bn(self.inv(out)))
#         out = self.avgpooltwo(out)+self.maxpooltwo(out)
#         out = self.relu(self.bn(self.inv(out)))
#         out = self.covpool1(out)
#         print("out", out.shape)
#         out = self.conv2d1(out)
#         print("out", out.shape)
        out = self.lip1(out)
        out = out.to(device)
        out1 = self.cross_attention(y)
        out1 = self.lip2(out1)
#         print("out1", out1.shape)
#         out1 = self.covpool2(out1)
#         out1 = self.conv2d2(out1)
        
        out2 = torch.cat((out,out1), 1)
#         out2 = self.bn(out2)
#         out2 = self.conv2d(out2)
        out3 = out2.view(out2.size(0), -1)
#         out3 = out.view(out.size(0), -1)
        out3 = self.dropout(out3)
        out4 = self.linear(out3)
        out5 = self.softmax(out4)
        return out5


# 定义类Args并实例化
class Args:
    def __init__(self,
                num_involution_reduct,
                num_involution_groups,
                involution_kernel_size,
                num_blocks,
                num_batch,
                patch_height,
                patch_width,
                num_patch_channels):
        self.num_involution_reduct = num_involution_reduct
        self.num_involution_groups = num_involution_groups
        self.involution_kernel_size = involution_kernel_size
        self.num_blocks = num_blocks
        self.num_batch = num_batch
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patch_channels = num_patch_channels
        
args = Args(2,2,[5,7,9],3,128,windowSize,windowSize,Bands)
# 实例化myNet
mynet = myNet(num_classes=16, channels=30, args=args)


# 定义优化器和损失函数
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mynet.parameters(),lr = 0.002, weight_decay=1e-03)


# 创建训练集数据库
BATCH_SIZE = 128
X_train = torch.tensor(X_train)
X_train = X_train.to(torch.float32)
X_train_permuted = X_train.permute(0,3,1,2)
print(X_train_permuted.size())

X_small_train = torch.tensor(X_small_train)
X_small_train = X_small_train.to(torch.float32)
X_small_train_permuted = X_small_train.permute(0,3,1,2)
print(X_small_train_permuted.size())

y_train = torch.tensor(y_train)
y_train = y_train.long()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu=torch.device("cpu")
mynet=mynet.to(device)
X_train_permuted=X_train_permuted.to(device)
X_small_train_permuted=X_small_train_permuted.to(device)
y_train=y_train.to(device)

train_dataset = torch.utils.data.TensorDataset(X_train_permuted, X_small_train_permuted, y_train)

train_loader = torch.utils.data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# 创建训练集数据库
BATCH_SIZE = 128
X_test = torch.tensor(X_test)
X_test = X_test.to(torch.float32)
X_test_permuted = X_test.permute(0,3,1,2)
# X_test_permuted=X_test_permuted.to(device)
print(X_test_permuted.size())

X_small_test = torch.tensor(X_small_test)
X_small_test = X_small_test.to(torch.float32)
X_small_test_permuted = X_small_test.permute(0,3,1,2)
print(X_small_test_permuted.size())

y_test = torch.tensor(y_test)
y_test = y_test.long()
# y_test=y_test.to(device)

test_dataset = torch.utils.data.TensorDataset(X_test_permuted, X_small_test_permuted, y_test)

test_loader = torch.utils.data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y, z in loader:
        x, y, z = x.to(device), y.to(device), z.to(device)
        with torch.no_grad():
            logits = model(x, y)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, z).sum().float().item()
    return correct / total


# 定义训练相关参数
epochs = 250
total_train_steps = 0  # 训练总次数
total_test_steps = 0  # 测试总次数

best_acc, best_epoch = 0, 0
global_step = 0

for epoch in range(epochs):
    print(f'-----第{epoch + 1}轮训练-----')
    epoch_total_train_losses = 0.0  # 每一轮训练总损失

#     mynet.train()  # 只对dropout等特定层有用
    # 开始训练
    for data in train_loader:
        features1, features2, targets = data
        mynet.train()  # 只对dropout等特定层有用
        output = mynet(features1, features2)
        loss = loss_func(output, targets)  # tensor(2.3019, grad_fn=<NllLossBackward>)
        # 优化器优化
        # ****1 梯度清零****
        optimizer.zero_grad()
        # ****2 计算梯度****
        loss.backward()
        # ****3 梯度更新****
        optimizer.step()
        global_step += 1
        
        total_train_steps += 1
        epoch_total_train_losses += loss.item()
        _,pre_lab = torch.max(output, 1)
        acc = accuracy_score(targets.cuda().data.cpu().numpy(), pre_lab.cuda().data.cpu().numpy())
        if total_train_steps % 10 == 0:
            print(f'第{total_train_steps}次训练 精度为：{acc.item()}')
            print(f'第{total_train_steps}次训练 损失为：{loss.item()}')

    print(f'-----第{epoch + 1}轮训练 训练集总损失{epoch_total_train_losses}-----')
    
    if epoch % 1 == 0:
            val_acc = evalute(mynet, test_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                print('current best acc:', best_acc, 'current best epoch:', best_epoch)
                torch.save(mynet.state_dict(), modelpath)
                
print('best acc:', best_acc, 'best epoch:', best_epoch)
mynet.load_state_dict(torch.load(modelpath))
print('loaded from ckpt!')


test_acc = evalute(mynet, test_loader)
print(test_acc)



# 4. Result

# # load best weights
# model-PC-SA.load_weights("./Salinas/model-PC-SA/best-model-PC-SA-Salinas-30-15.hdf5")
# model-PC-SA.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Y_pred_test = model-PC-SA.predict([X_test, X_small_test, SBNL_test])# , Xtest
# y_pred_test = np.argmax(Y_pred_test, axis=1) # np.argmax用于返回一个numpy数组中最大值的索引值，因为之前将y向量转化为二值类别矩阵，故最大值（1）下标即为类别（之前将向量元素减1，故可以对应上）
# oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
# kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_test)
# classification = classification_report(np.argmax(y_test, axis=1), y_pred_test)
# print(classification, oa, kappa)


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def reports (loader,y_test,name):
    mynet.load_state_dict(torch.load(modelpath))
    mynet.eval()
    count = 0
    for x, y, z in loader:
        x, y, z = x.to(device), y.to(device), z.to(device)
        with torch.no_grad():
            logits = mynet(x, y)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().numpy()
        if count == 0:
            y_pred = pred
            count = 1
        else:
            y_pred = np.concatenate((y_pred, pred))
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'PC':
        target_names = ['Water','Trees','Asphalt','Self-Blocking Bricks', 'Bitumen','Tiles','Shadows',
                        'Meadows','Bare Soil']
    elif name == 'H13':
        target_names = ['Healthy_grass', 'Stressed_grass_notill', 'Synthetic_grass', 'Trees'
                        ,'Soil', 'Water', 'Residential', 
                        'Commercial', 'Road', 'Highway', 'Railway',
                        'Parking_lot_1', 'Parking_lot_2 ', 'Tennis_court ', 'Running_track ',]
    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100

Y_test = y_test.numpy()
classification, confusion, oa, each_acc, aa, kappa = reports(test_loader,Y_test,dataset)
classification = str(classification)
confusion = str(confusion)
# os.mkdir("./Salinas/output")# 创建目录
file_name = resultpath

with open(file_name, 'w') as x_file:
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(each_acc))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))
print('Write complete!')

# load the original image
X, y = loadData(dataset)

height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
SMALL_PATCH_SIZE = smallwindowSize

X_pca, pca = applyPCA(X, numComponents=Bands)

X_pca = padWithZeros(X_pca, PATCH_SIZE // 2)
X = padWithZeros(X, SMALL_PATCH_SIZE // 2)


def Patch(data, height_index, width_index):
    # transpose_array = data.transpose((2,0,1))
    # print transpose_array.shape
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]

    return patch


def SmallPatch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + SMALL_PATCH_SIZE)
    width_slice = slice(width_index, width_index + SMALL_PATCH_SIZE)
    smallpatch = data[height_slice, width_slice, :]

    return smallpatch


# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        if target == 0:
            continue
        else:
            image_patch = Patch(X_pca, i, j)
            X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                               image_patch.shape[2])#.astype('float32')
            X_test_image = torch.FloatTensor(X_test_image.transpose(0,3,1,2)).to(device)

            imagesmall_patch = SmallPatch(X, i, j)
            Xsmall_test_image = imagesmall_patch.reshape(1, imagesmall_patch.shape[0], imagesmall_patch.shape[1],
                                                         imagesmall_patch.shape[2])#.astype('float32')
            Xsmall_test_image = torch.FloatTensor(Xsmall_test_image.transpose(0,3,1,2)).to(device)

            # mynet.eval()
            # with torch.no_grad():
            logits = mynet(X_test_image, Xsmall_test_image)
            prediction = (logits.argmax(dim=1)).cpu().numpy()
            outputs[i][j] = prediction + 1

# plt.figure(figsize=(10, 8))
# plt.imshow(outputs.astype(int),cmap='nipy_spectral')
# plt.colorbar()
# plt.axis('off')
# plt.savefig('IP_cmap.png')
# plt.show()

# Plot the Ground Truth Image
ground_truth = spectral.imshow(classes=y, figsize=(7, 7))

# Plot the Predicted image
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))

spectral.save_rgb(imagepath, outputs.astype(int), colors=spectral.spy_colors)
