# -*- coding: utf-8 -*-

def cut_pic(door_img, pts_doorbox):
    # pts_doorbox = np.array([xmin, ymin, xmax, ymax)

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)
    # height,width = door_img.shape[:2]
    # print(len_W,len_H,len_rec_1)
    # time.sleep(1000)
    pic_xmin= int(len_rec_1/4)
    pic_ymin = 0
    # print(pic_xmin,pic_ymin)  #564 14
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin, pic_ymin, len_rec_1_reset)
    # return (crop_door_img_1,pic_xmin,pic_ymin,)
    # # return (crop_door_img_2,minX_temp_2,minY_temp_2,width,height,len_rec_2)
def cut_pic_incase_stickLocation_is_low(door_img,pts_doorbox):

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)

    pic_xmin= int(len_rec_1/4)
    # pic_ymin = int(pts_doorbox[1])
    pic_ymin = int(len_H/4)
    # print(pic_xmin,pic_ymin)  #564 14
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin,pic_ymin,len_rec_1_reset)


def cut_pic_incase_stickLocation_is_low_1(door_img,pts_doorbox):

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)

    pic_xmin= int(len_rec_1/4)
    # pic_ymin = int(pts_doorbox[1])
    pic_ymin = int(len_H/5)
    # print(pic_xmin,pic_ymin)  #564 14
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin,pic_ymin,len_rec_1_reset)


def cut_pic_incase_stickLocation_is_low_2(door_img,pts_doorbox):

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)

    pic_xmin= int(len_rec_1/4)
    pic_ymin = int(2*len_H/5)
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin,pic_ymin,len_rec_1_reset)

def cut_pic_incase_stickLocation_is_Left(door_img,pts_doorbox):

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)

    pic_xmin= 0
    pic_ymin = 0
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin,pic_ymin,len_rec_1_reset)

def cut_pic_incase_stickLocation_is_Right(door_img,pts_doorbox):

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)

    pic_xmin= int(len_rec_1/2)
    pic_ymin = 0
    len_rec_1_reset= len_rec_1/2

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin,pic_ymin,len_rec_1_reset)

#
def cut_pic_big(door_img, pts_doorbox):
    # pts_doorbox = np.array([xmin, ymin, xmax, ymax)

    len_W = pts_doorbox[2]-pts_doorbox[0]
    len_H = pts_doorbox[3]-pts_doorbox[1]
    len_rec_1 = min(len_W, len_H)
    # height,width = door_img.shape[:2]
    # print(len_W,len_H,len_rec_1)
    # time.sleep(1000)
    pic_xmin= 0
    pic_ymin = 0
    len_rec_1_reset= len_rec_1

    crop_door_img_1 = door_img[pic_ymin:int((pic_ymin +len_rec_1_reset)), pic_xmin:int((pic_xmin +len_rec_1_reset))]

    return (crop_door_img_1, pic_xmin, pic_ymin, len_rec_1_reset)
    # # return (crop_door_img_2,minX_temp_2,minY_temp_2,width,height,len_rec_2)