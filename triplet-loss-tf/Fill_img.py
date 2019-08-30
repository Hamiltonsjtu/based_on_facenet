#_*_ coding: utf-8 _*_
'''给图片外围填充黑色像素'''
import cv2
import numpy as np
import os

dirpath = r'F:\110105036-2126-5505_OK'
dirpath_fill = r'F:\110105036-2126-5505_OK_fill'
os.makedirs(dirpath_fill, exist_ok = True)

for i in os.listdir(dirpath):
    i_path = dirpath + '/' + i
    os.makedirs(dirpath_fill + '/' + i, exist_ok=True)
    for j in os.listdir(i_path):
        img_path = i_path + '/' + j
        img = cv2.imread(img_path)
        height, width, chn = img.shape[0], img.shape[1], img.shape[2]
        max_h_w = max(height, width)
        ratio = 300/max_h_w
        h_n = int(height*ratio)
        w_n = int(width*ratio)
        img = cv2.resize(img, (w_n, h_n))
        # center_h = h_n/2
        # center_w = w_n/2
        top = int(np.floor((300-h_n)/2))
        bottom = int(np.ceil((300-h_n)/2))
        left = int(np.floor((300-w_n)/2))
        right = int(np.ceil((300-w_n)/2))
        img_n = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        savepath = dirpath_fill + '/' + i + '/' + j
        cv2.imwrite(savepath, img_n)

# for file in os.listdir(dirpath1):
#     img = cv2.imread(dirpath1+os.sep+file)
#     if float(img.shape[0]/img.shape[1]) > aspect_rate:
#         tem_extend = int(((img.shape[0]/aspect_rate)-img.shape[1])/2)
#         # print("水平扩展区域大小",2*tem_extend*img.shape[1])
#         fill_img = cv2.copyMakeBorder(img,0,0,tem_extend,tem_extend, cv2.BORDER_CONSTANT,value=[0,0,0])
#         print(fill_img.shape[0]/fill_img.shape[1])
#     else:
#         tem_extend = int((img.shape[1]*aspect_rate-img.shape[0])/2)
#         # print("竖直扩展区域大小",2*tem_extend * img.shape[0])
#         fill_img = cv2.copyMakeBorder(img,tem_extend,tem_extend,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])
#         print(fill_img.shape[0]/fill_img.shape[1])

    # cv2.namedWindow('image1', 0)
    # cv2.imshow('image1', fill_img)
    # img = cv2.resize(fill_img,input_size)
    # cv2.imwrite(dirpath2+os.sep+file,fill_img)