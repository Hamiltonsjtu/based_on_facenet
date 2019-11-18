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
    head_ytl_u, head_ybr_u, head_xtl_u, head_xbr_u = oriBox

    boxImg = oriImage[head_ytl_u: head_ybr_u, head_xtl_u: head_xbr_u, :]
    boxImgHeight, boxImgWidth, _ = np.shape(boxImg)
    newBox_width = NewBox[0]
    newBox_height = NewBox[1]
    w_ratio = float(boxImgWidth)/float(newBox_width)
    h_ratio = float(boxImgHeight)/float(newBox_height)

    ratio = np.maximum(w_ratio, h_ratio)
    w_tmp = int(1.0/ratio*boxImgWidth)
    h_tmp = int(1.0/ratio*boxImgHeight)
    image = cv2.resize(boxImg, (w_tmp, h_tmp))
    top = newBox_height - h_tmp
    right = newBox_width - w_tmp
    bottom = 0
    left = 0
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, boxImg


# img_cls = [img_dir + '/' + i for i in os.listdir(img_dir) if os.path.isdir(img_dir + '/' + i)]

# # df = pd.read_table([img_dir + '/' + i for i in os.listdir(img_dir) if i.endswith('txt')][0])
# anno_txt_1 = r'F:\FocusScreen\FocusImage\not_Duplicated_img.txt'
# df_h1 = pd.read_csv(anno_txt_1, sep = ',')
# col_n = ['name', 'head_xtl', 'head_ytl', 'head_xbr', 'head_ybr', 'label', 'width', 'height', 'screen_xtl', 'screen_ytl', 'screen_xbr', 'screen_ybr']
# df_h1 = pd.DataFrame(df_h1, columns=col_n)
# df_h1.rename(columns={'head_xtl':'s_xtl','head_ytl':'s_ytl', 'head_xbr':'s_xbr', 'head_ybr':'s_ybr', 'screen_xtl':'h_xtl', 'screen_ytl':'h_ytl', 'screen_xbr':'h_xbr', 'screen_ybr':'h_ybr'}, inplace=True)
# df_h1.rename(columns={'s_xtl':'screen_xtl','s_ytl':'screen_ytl', 's_xbr':'screen_xbr', 's_ybr':'screen_ybr', 'h_xtl':'head_xtl', 'h_ytl':'head_ytl','h_xbr':'head_xbr', 'h_ybr':'head_ybr'}, inplace=True)
#
# anno_txt_2 = r'F:\FocusScreen\eye_yes_unsure.txt'
# df_h2 = pd.read_csv(anno_txt_2, header=None, sep='\t')
# # df_h2 = pd.DataFrame(df_h2,columns=col_n)
# df_h2 = df_h2.iloc[0:-1,0:12]
# df_h2.columns=['name', 'width', 'height', 'screen_xtl', 'screen_ytl', 'screen_xbr', 'screen_ybr', 'head_xtl','head_ytl', 'head_xbr', 'head_ybr', 'label']
#
# columns_list = list(df_h1.columns)
# df_1 = df_h1[columns_list]
# df_2 = df_h2[columns_list]
# df = df_1.append(df_2)
# # # fm = open(r'./Images_Anno.txt', 'a+')
# # df.to_csv(r'./Image_Anono.csv')

df = pd.read_csv(r'F:\FocusScreen\eye_all.txt', header = None,sep=',')
df.columns = ['name', 'screen_xtl', 'screen_ytl', 'screen_xbr', 'screen_ybr', 'label','width', 'height', 'head_xtl', 'head_ytl', 'head_xbr', 'head_ybr', ]
# df_up = df.loc[df['label'].isin(['no', 'yes', 'unsure'])]
# df_up.to_csv(r'./Image_Anono_up.csv')
img_dir = r'F:\FocusScreen\Images'

img_dst = r'F:\FocusScreen\Headmages'
if not os.path.exists(img_dst):
    os.mkdir(img_dst)
ann_dst = r'F:\FocusScreen\Img_anno.csv'
ann_data = []
img_paths = os.listdir(img_dir)
for k in img_paths:
    # original image shape
    # k = '2019_10_07_00_30_38_D34308405_310104017-4313-9694.jpg'
    img_path = img_dir + '/' + k
    image = cv2.imread(img_path)
    ori_h, ori_w, c = image.shape
    ## we recommend the ratio of width/height is 0.5625,
    # namely the ratio of 720*1280 or 1080*1920
    w = 720
    h = 1280
    img_loc_data = df
    if len(df[df['name'] == k]) != 0:
        for i in range(len(df[df['name'] == k])):
            head_xtl = float(list(df[df['name'] == k]['head_xtl'])[i])
            head_ytl = float(list(df[df['name'] == k]['head_ytl'])[i])
            head_xbr = float(list(df[df['name'] == k]['head_xbr'])[i])
            head_ybr = float(list(df[df['name'] == k]['head_ybr'])[i])

            screen_xtl = float(list(df[df['name'] == k]['screen_xtl'])[i])
            screen_ytl = float(list(df[df['name'] == k]['screen_ytl'])[i])
            screen_xbr = float(list(df[df['name'] == k]['screen_xbr'])[i])
            screen_ybr = float(list(df[df['name'] == k]['screen_ybr'])[i])

            img_n, head_xbr_u, head_ybr_u = point_relocate(image, ori_w, ori_h, w, h, head_xbr, head_ybr)
            _, head_xtl_u, head_ytl_u = point_relocate(image, ori_w, ori_h, w, h, head_xtl, head_ytl)
            _, screen_xbr_u, screen_ybr_u = point_relocate(image, ori_w, ori_h, w, h, screen_xbr, screen_ybr)
            _, screen_xtl_u, screen_ytl_u = point_relocate(image, ori_w, ori_h, w, h, screen_xtl, screen_ytl)

            head = [head_xtl_u, head_ybr_u, head_xbr_u, head_ytl_u]
            screen = [screen_xtl_u, screen_ybr_u, screen_xbr_u, screen_ytl_u]

            head_img_fill, head_img = UnstructBox_Square(img_n, [head_ytl_u,head_ybr_u, head_xtl_u,head_xbr_u], (255,255))

            # cv2.imshow('original_head', head_img)
            # cv2.imshow('fill_head', head_img_fill)
            k_name, extension = os.path.splitext(k)
            head_img_path = img_dst + '/' + k_name + '_' + str(i) + extension
            cv2.imwrite(head_img_path, head_img_fill)
            pair_label = list(df[df['name'] == k]['label'])[i]
            single = [k_name + '_' + str(i) + extension, screen_xtl_u, screen_ybr_u, screen_xbr_u, screen_ytl_u, pair_label, head_xtl, head_ybr, head_xbr, head_ytl]
            ann_data.append(single)

            # image = cv2.rectangle(image, (int(head_xtl), int(head_ybr)), (int(head_xbr), int(head_ytl)), (0,0,255))
            # orig_img = cv2.rectangle(image, (int(screen_xtl), int(screen_ybr)), (int(screen_xbr), int(screen_ytl)), (0,255,0))
            # txt_img_ = cv2.putText(orig_img, str(i) + '_' + str(list(df[df['name'] == k]['label'])[i]), (int(head_xtl), int(head_ybr)), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            # cv2.imshow('original_' + k_name, txt_img_)
            #
            # img_n = cv2.rectangle(img_n, (int(screen_xtl_u), int(screen_ybr_u)), (int(screen_xbr_u), int(screen_ytl_u)), (0,255,0))
            # reshape_img = cv2.rectangle(img_n, (int(head_xtl_u), int(head_ybr_u)), (int(head_xbr_u), int(head_ytl_u)), (0,0,255))
            # txt_img = cv2.putText(reshape_img, str(i) + '_' + str(list(df[df['name'] == k]['label'])[i]), (int(head_xtl_u), int(head_ybr_u)), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            # cv2.imshow('resized_' + k_name, txt_img)
            #
            # cv2.waitKey(3000)
            # cv2.destroyAllWindows()
    else:
        print('-----  image is not in ano.txt  ----')
        continue

pd_update = pd.DataFrame(ann_data)
pd_update.columns = ['name', 'screen_xtl', 'screen_ybr', 'screen_xbr', 'screen_ytl', 'pair_label', 'head_xtl', 'head_ybr', 'head_xbr', 'head_ytl']
# pd_update.rename(columns={0:'name', 1: 'head_xtl',
#                           2: 'head_ybr', 3: 'head_xbr',
#                           4: 'head_ytl', 5: 'pair_label',
#                           6: 'screen_xtl_u', 7: 'screen_ybr_u',
#                           8: 'screen_xbr_u', 9: 'screen_ytl_u'}, inplace = True)
pd_update.to_csv(r'./Head_Anono.csv')

