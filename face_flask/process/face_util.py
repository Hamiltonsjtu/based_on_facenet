#coding:utf-8
import cv2
import numpy as np
from scipy import misc


def mtcnn_crop_face_img(img, bboxes):
    face_imgs_list = []
    face_crop_margin = 10
    face_size = 160
    for index, bbox in enumerate(bboxes):
        bbox_accuracy = bbox[4] * 100.0
        if bbox_accuracy < 90:
            continue

        w, h = img.shape[0:2]
        left = int(np.maximum(bbox[0] - face_crop_margin / 2, 0))
        top = int(np.maximum(bbox[1] - face_crop_margin / 2, 0))
        right = int(np.minimum(bbox[2] + face_crop_margin / 2, h))
        bottom = int(np.minimum(bbox[3] + face_crop_margin / 2, w))

        cropped = img[top:bottom, left:right, :]
        cropped_img = cv2.resize(cropped, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
        face_imgs_list.append(cropped_img)
    return face_imgs_list


def crop_ssd_face_img(img, bboxes, filename):
    face_imgs_list = []
    for index, bbox in enumerate(bboxes):
        left = int(bbox[0])
        top = int(bbox[1])
        right = int(bbox[2])
        bottom = int(bbox[3])
        cropped = img[top:bottom, left:right, :]
        face_imgs_list.append(cropped)
        #cv2.imwrite('C:\\Users\\kc\\Desktop\\face\\face_'+filename+str(index) + '.jpg', cropped)
    return face_imgs_list


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def crop_ssd_face_img_update(img, bboxes, filename):
    ###
    # margin can be given as 0.15
    # img_size: the size of img
    # image_size: the size of cropped face, recommend 160
    ###

    margin = 0.15
    img_size = np.asarray(img.shape)[0:2]
    image_size = 160
    face_imgs_list = []
    bb_new_list = []
    for _, det in enumerate(bboxes):
        bb = np.zeros(4, dtype=np.int32)
        if margin > 1:
            # =================================================
            # cropped with fixed margin which is used for lfw
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[2] - margin / 2, 0)
            bb[2] = np.minimum(det[1] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        else:
            # =================================================
            # cropped with percentage margin can be used for images download from internet
            width = det[1] - det[0]
            height = det[3] - det[2]
            bb[0] = np.maximum(det[0] - margin * width, 0)
            bb[1] = np.maximum(det[2] - margin * height, 0)
            bb[2] = np.minimum(det[1] + margin * width, img_size[1])
            bb[3] = np.minimum(det[3] + margin * height, img_size[0])
        print('ssd return window height: {}, width: {}'.format(bb[3] - bb[1], bb[2] - bb[0]))

        #### crop by square box
        center = [float(bb[0] + bb[2]) / 2, float(bb[1] + bb[3]) / 2]
        height = float(bb[3] - bb[1])
        weight = float(bb[2] - bb[0])
        max_size = max([height, weight]) / 2
        img_height = img.shape[0]
        img_width = img.shape[1]
        size_up = img_height - center[1]
        size_down = center[1]
        size_left = center[0]
        size_right = img_width - center[0]

        print('adjust size ', [max_size, size_up, size_down, size_left, size_right])
        adjust_size = min([max_size, size_up, size_down, size_left, size_right])
        bb_new = np.array(
            [center[0] - adjust_size, center[1] - adjust_size, center[0] + adjust_size, center[1] + adjust_size],
            dtype=np.int32)
        print('crop window height: {}, width: {}'.format(bb_new[3] - bb_new[1], bb_new[2] - bb_new[0]))
        cropped = img[bb_new[1]:bb_new[3], bb_new[0]:bb_new[2], :]

        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        print('cropped image shape: {}'.format(np.shape(scaled)))
        prewhitened = prewhiten(scaled)
        face_imgs_list.append(prewhitened)
        bb_new_list.append(bb_new)

    return face_imgs_list, bb_new_list
