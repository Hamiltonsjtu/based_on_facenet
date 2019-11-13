"""
Reshape image with size (w, h) in ton (w_n, h_n)
and relocate the bounding box.
"""

import numpy as np
import cv2
import os
import pandas as pd

def point_relocate(image, ori_w, ori_h, w, h, point_x, point_y):
    w_ratio = float(w)/float(ori_w)
    h_ratio = float(h)/float(ori_h)
    ratio = np.minimum(w_ratio, h_ratio)
    w_tmp = int(ratio*ori_w)
    h_tmp = int(ratio*ori_h)
    image = cv2.resize(image, (w_tmp, h_tmp))
    point_x = int(ratio*point_x)
    point_y = int(ratio*point_y)
    top = h - h_tmp
    right = w - w_tmp
    bottom = 0
    left = 0
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, point_x, point_y

def UnstructBox_Square(oriImage, oriBox, NewBox):
    """
    Auhor: Hamilton
    :param oriImage: input image
    :param oriBox: left-top (ltx, lty), right-bottom (rbx, rby)
    :param NewBox: new box size, for example (160, 250)
    :return: resized square box with copyMakeBorder
    """
    ltx = oriBox[0]
    lty = oriBox[1]
    rbx = oriBox[2]
    rby = oriBox[3]
    boxImg = oriImage[ltx:rbx, rby:lty, :]
    boxImgHeight, boxImgWidth, _ = boxImg.shape()
    newBox_width = NewBox[0]
    newBox_height = NewBox[1]
    w_ratio = float(boxImgWidth)/float(newBox_width)
    h_ratio = float(boxImgHeight)/float(newBox_height)
    ratio = np.minimum(w_ratio, h_ratio)
    w_tmp = int(ratio*w_ratio)
    h_tmp = int(ratio*h_ratio)
    image = cv2.resize(boxImg, (w_tmp, h_tmp))
    top = newBox_height - h_tmp
    right = newBox_width - w_tmp
    bottom = 0
    left = 0
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


img_dir = r'F:\FocusScreen\FocusImage'
img_cls = [img_dir + '/' + i for i in os.listdir(img_dir) if os.path.isdir(img_dir + '/' + i)]

# df = pd.read_table([img_dir + '/' + i for i in os.listdir(img_dir) if i.endswith('txt')][0])
df = pd.read_csv([img_dir + '/' + i for i in os.listdir(img_dir) if i.endswith('txt')][0], sep = '\t')



for i in img_cls:
    img_path = [i + '/' + j for j in os.listdir(i)]
    for k in img_path:
        # original image shape
        # k = r'F:\FocusScreen\FocusImage\yes\2019_10_07_00_30_19_225039882_330110005-2754-8338.jpg'
        image = cv2.imread(k)
        ori_h, ori_w, c = image.shape
        ## we recommend the ratio of width/height is 0.5625,
        # namely the ratio of 720*1280 or 1080*1920
        w = 720
        h = 1280

        print('df len: ', len(df))
        head_xtl = df['head_xtl']
        head_ytl = df['head_ytl']
        head_xbr = df['head_xbr']
        head_ybr = df['head_ybr']

        screen_xtl = df['screen_xtl']
        screen_ytl = df['screen_ytl']
        screen_xbr = df['screen_xbr']
        screen_ybr = df['screen_ybr']

        img_n, head_xbr_u, head_ybr_u = point_relocate(image, ori_w, ori_h, w, h, head_xbr, head_ybr)
        _, head_xtl_u, head_ytl_u = point_relocate(image, ori_w, ori_h, w, h, head_xtl, head_ytl)
        _, screen_xbr_u, screen_ybr_u = point_relocate(image, ori_w, ori_h, w, h, screen_xbr, screen_ybr)
        _, screen_xtl_u, screen_ytl_u = point_relocate(image, ori_w, ori_h, w, h, screen_xtl, screen_ytl)

        head = [head_xtl_u, head_ybr_u, head_xbr_u, head_ytl_u]
        screen = [screen_xtl_u, screen_ybr_u, screen_xbr_u, screen_ytl_u]

        # tect_img = cv2.rectangle(image, (int(head_xtl), int(head_ybr)), (int(head_xbr), int(head_ytl)), (0,0,255))
        # cv2.imshow('testRect', tect_img)
        image = cv2.rectangle(image, (int(head_xtl), int(head_ybr)), (int(head_xbr), int(head_ytl)), (0,0,255))
        orig_img = cv2.rectangle(image, (int(screen_xtl), int(screen_ybr)), (int(screen_xbr), int(screen_ytl)), (0,255,0))
        cv2.imshow('original', orig_img)

        img_n = cv2.rectangle(img_n, (int(screen_xtl_u), int(screen_ybr_u)), (int(screen_xbr_u), int(screen_ytl_u)), (0,255,0))
        reshape_img = cv2.rectangle(img_n, (int(head_xtl_u), int(head_ybr_u)), (int(head_xbr_u), int(head_ytl_u)), (0,0,255))
        cv2.imshow('resized',reshape_img)



        cv2.waitKey()
        cv2.destroyAllWindows()