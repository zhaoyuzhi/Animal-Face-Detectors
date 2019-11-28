# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:36:57 2018

@author: yzzhao2
"""

import numpy as np
import cv2
import os

cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_ext_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

def catface_detector(imgpath, SF = 1.1, N = 3):
    # read the image
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cats = cat_cascade.detectMultiScale(gray, scaleFactor = SF, minNeighbors = N)
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor = SF, minNeighbors = N)
    
    for (x, y, w, h) in cats:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in cats_ext:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img

def catface_generator(imgpath, savepath, SF = 1.1, N = 3):
    # read the image
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cats = cat_cascade.detectMultiScale(gray, scaleFactor = SF, minNeighbors = N)
    
    for i, (x, y, w, h) in enumerate(cats):
        img = img[y : (y + h), x : (x + w), :]
        savename =  str(i) + imgpath.split('\\')[-1]
        savename = savepath + '\\' + savename
        cv2.imwrite(savename, img)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

if __name__ == "__main__":

    '''
    # This part is for ImageNet
    basepath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\ILSVRC2012_train_256'
    savepath = './catface_ImageNet'
    imglist = get_files(basepath)
    ImageNet_cat_range = [2085620, 2138441]
    for i, imgpath in enumerate(imglist):
        if i % 1000 == 0:
            print('Now it is the %d-th image', i)
        imgnumber = int(imgpath.split('\\')[-1][2:9])
        if imgnumber >= ImageNet_cat_range[0] and imgnumber <= ImageNet_cat_range[1]:
            catface.catface_generator(imgpath, savepath)
    '''

    '''
    This part is for CUHK cat face dataset
    basepath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\cat_dataset'
    savepath = './catface_CUHKcat'
    imglist = get_files(basepath)
    for i, imgpath in enumerate(imglist):
        if imgpath.split('\\')[-1][-3:] == 'jpg':
            if i % 1000 == 0:
                print('Now it is the %d-th image', i)
            catface.catface_generator(imgpath, savepath)
    '''
    
    '''
    # This part is for Columbia dog dataset
    basepath = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\CU_Dogs\\dogImages'
    savepath = './dogface_CUdog'
    imglist = get_files(basepath)
    for i, imgpath in enumerate(imglist):
        if i % 1000 == 0:
            print('Now it is the %d-th image', i)
        catface_generator(imgpath, savepath)
    '''

    # Image-based experiment
    img = catface_detector('./cat1.jpg')
    cv2.imwrite('./cat1_result.jpg', img)
    cv2.imshow('detected image', img)
    cv2.waitKey(0)
