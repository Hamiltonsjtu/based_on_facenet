# import os
import shutil
# import cv2


#_*_ coding: utf-8 _*_
'''给图片外围填充黑色像素'''
import cv2
import numpy as np
import os

## remove dir which contains less 2 images
# dirpath_fill = r'F:\SAVE_PIC_1_fill'
# peo_dir = os.listdir(dirpath_fill)
# for i in peo_dir:
#     ela_path = os.path.join(dirpath_fill, i)
#     peo_paths = os.listdir(ela_path)
#     for j in peo_paths:
#         peo_path = ela_path + '/' + j
#         img_path = os.listdir(peo_path)
#         img_num = len(img_path)
#         if img_num < 2:
#             print(peo_path)
#             for j in img_path:
#                 img_all_path = os.path.join(peo_path, j)
#                 print('---- delete file ----')
#                 os.remove(img_all_path)
#             print('---- delete empty folder ----')
#             os.removedirs(peo_path)
#
# def is_chinese(uchar):
#     """判断一个unicode是否是汉字"""
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
#

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



scr = r'F:\NotBusiness'
dst = r'F:\NotBusiness_fill'
dirpath = dst
dirpath_fill = dst

# scr = r'F:\OF_TEST'
# dst = r'F:\OF_TEST_fill'
# dirpath = dst
# dirpath_fill = dst

# if not os.path.exists(dst):
#     os.mkdir(dst)
# sub_dir = os.listdir(scr)
#
# for i in sub_dir:
#     sub_dir_full = scr + '/' + i
#     peos = os.listdir(sub_dir_full)
#     for j in peos:
#         peo_dir = sub_dir_full + '/' + j
#         print(peo_dir)

# for i in sub_dir:
#     sub_dir_full = scr + '/' + i
#     peos = os.listdir(sub_dir_full)
#
#     for j in peos:
#         peo_dir = sub_dir_full + '/' + j
#         if os.path.isdir(peo_dir):
#             dst_peo = sub_dir_full + '/' + i[:-3] + '_' + j
#             os.rename(peo_dir, dst_peo)
#             for k in os.listdir(dst_peo):
#                 img_scr = dst_peo + '/' + k
#                 img_dst = dst_peo + '/' + i[:-3] + '_' + k
#                 os.rename(img_scr, img_dst)
#         else:
#             print('=== delete not folder files ===')
#             os.remove(peo_dir)

# for i in sub_dir:
#     sub_dir_full = os.path.join(scr, i)
#     peos = os.listdir(sub_dir_full)
#     for j in peos:
#         sub_sub_dir_full = os.path.join(sub_dir_full, j)
#         subsub2sub = os.path.join(dst, i+'_'+j)
#         if not os.path.exists(subsub2sub):
#             os.mkdir(subsub2sub)
#         # os.system('scp sub_sub_dir_full subsub2sub')
#         for k in os.listdir(sub_sub_dir_full):
#             if k.endswith('.db'):
#                 continue
#             else:
#                 img_path = os.path.join(sub_sub_dir_full, k)
#                 img_subsub2sub = os.path.join(subsub2sub, k)
#                 if os.path.isdir(img_path):
#                     continue
#                 else:
#                     shutil.copy2(img_path, img_subsub2sub, follow_symlinks=True)


####   RESHAPE AND FILL THE RAW IMAGE    ####


os.makedirs(dirpath_fill, exist_ok = True)

for i in os.listdir(scr):
    i_path = scr + '/' + i
    os.makedirs(scr + '/' + i, exist_ok=True)
    for j in os.listdir(i_path):
        if j == '其他':
            continue
        elif j.endswith('.db'):
            continue
        # elif j.endswith('jpg'):
        #     continue
        else:
            peo_path = i_path + '/' + j
            # os.makedirs(dirpath_fill + '/' + i + '/' + j, exist_ok=True)
            for k in os.listdir(peo_path):
                if k.endswith('jpg') or k.endswith('png'):
                    img_path = peo_path
                    img = cv2.imread(os.path.join(img_path, k))
                    if len(np.shape(img)) == 0:
                        continue
                    else:
                        print('img_path: {} and type{}'.format(img_path, type(img)))
                        height, width, chn = img.shape[0], img.shape[1], img.shape[2]
                        max_h_w = max(height, width)
                        dimsize = 600
                        ratio = dimsize/max_h_w
                        h_n = int(height*ratio)
                        w_n = int(width*ratio)
                        img = cv2.resize(img, (w_n, h_n))
                        # center_h = h_n/2
                        # center_w = w_n/2
                        top = int(np.floor((dimsize-h_n)/2))
                        bottom = int(np.ceil((dimsize-h_n)/2))
                        left = int(np.floor((dimsize-w_n)/2))
                        right = int(np.ceil((dimsize-w_n)/2))
                        img_n = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        img_n = cv2.resize(img_n,(dimsize, dimsize))
                        savepath_file = dirpath_fill + '/' + i + '/' + j
                        os.makedirs(savepath_file, exist_ok=True)

                        savepath = dirpath_fill + '/' + i + '/' + j + '/' + k

                        cv2.imwrite(savepath, img_n)
                else:
                    continue

#
# import tensorflow as tf
#
# scr_1 = r'F:\dd\110105036-2126-5505_8\hujintao_0000.jpg'
# scr_2 = r'E:\train_TEST\fff\OK2_OK2_84\104083.jpg'
# img_1 = tf.image.decode_image(scr_1, dtype=tf.uint8)
# img_2 = tf.image.decode_image(scr_2)
#
# with tf.Session() as sess:
#     sess.run(img_1)
#     sess.run(img_2)
#     print('OK!')
# img_1 = cv2.imread(scr_1)
# img_2 = cv2.imread(scr_2)
#
# print('img_1 and img_2 shape is {}--{}'.format(img_1.shape, img_2.shape))
