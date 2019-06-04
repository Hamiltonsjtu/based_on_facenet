from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("src")                  # FACENET
import facenet
import align.detect_face

sys.path.append('SSD_face_detection')   # SSD
from SSD_face_detection.SSD_detect_face import ssd_find_face as SSD

import dlib
from scipy import misc
import tensorflow as tf
import numpy as np
import time
import sys
import os
import copy
import argparse
import scipy.stats as st
import cv2
import open_Face

# def main(args):
img_path = 'xi_hu.jpg'  #args.image_path
print(os.path.splitext(img_path))



# ################## =======  detect face by SSD   ====== ###################s
# start_time = time.time()
# boxes, scores, face_indice, im_height, im_width, image = SSD(img_path)
# det_arry = boxes[face_indice, :]
# scores_arr = scores[face_indice]
# print('type of boxes: {} and its value: {}'.format(type(boxes), boxes))
# print('SSD detect face cost {}s'.format(time.time()-start_time))

################## =======  detect face by MTRCNN  ====== ###################
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
print('Creating MTRCNN networks and loading parameters')
start_time = time.time()
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

image = misc.imread(img_path, mode='RGB')
# image = cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB)
img_size = np.asarray(image.shape)[0:2]
boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
face_indice = np.arange(np.shape(boxes)[0])
print('type of boxes: {} and its value: {}'.format(type(boxes), boxes))
print('MTRCNN detect face cost {}s'.format(time.time()-start_time))
boxes_ = np.zeros_like(boxes[:,0:-1], dtype='int')
boxes_[:,0] = boxes[:,2]
boxes_[:,1] = boxes[:,0]
boxes_[:,2] = boxes[:,1]
boxes_[:,3] = boxes[:,3]
scores = boxes[:,-1]
boxes = boxes_
print('type of boxes: {} and its value: {}'.format(type(boxes), boxes))

##############################################################################
# #####  openCV SHOW IMAGE AND CROPPED AREA
# for i in face_indice:
#     # print('the {}-th face'.format(i))
#     print('{}# box: {} its type {} and score {}'.format(i, boxes[i,:], type(boxes[i,0]), scores[i]))
#     box = boxes[i, :]
#     print('box {} and its type {}'.format(box, type(box)))
#     img_crop = cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (0,255,0))
#     cv2.imshow('img_crp', img_crop)
# cv2.waitKey()

##### =======  landmark and project for faces  ====== #######
predictor_model = "dlib/shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = open_Face.AlignDlib(predictor_model)

win = dlib.image_window()
win.set_image(image)

for i in face_indice:
    box = boxes[i, :]
    xmin = box[0]
    xmax = box[1]
    ymin = box[2]
    ymax = box[3]

    print('the {}# face: xmin {}, ymin {}, xmax {}, ymax {}'.format(i, xmin, ymin, xmax, ymax))

    det_arry_dlib = dlib.rectangle(left = int(xmin), top = int(ymin), right = int(xmax), bottom = int(ymax))
    print('type of det_arr_dlib: {} and value: {}'.format(type(det_arry_dlib),det_arry_dlib))
    # Draw a box around each face we found
    win.add_overlay(det_arry_dlib)
    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, det_arry_dlib)
    # print('=========================')
    # print(pose_landmarks)
    # Use openface to calculate and perform the face alignment
    alignedFace = face_aligner.align(534, image, det_arry_dlib, landmarkIndices=open_Face.AlignDlib.OUTER_EYES_AND_NOSE)
    cv2.imwrite("xi_hu{}.jpg".format(i), alignedFace)

    win.add_overlay(pose_landmarks)
dlib.hit_enter_to_continue()


# for i in face_indice:
#     # print('the {}-th face'.format(i))
#     # print('{} box: {} and score {}'.format(i, boxes[i,:], scores[i]))
#     box = boxes[i, :]
#     ymin = box[0]
#     xmin = box[1]
#     ymax = box[2]
#     xmax = box[3]
#     bb = np.zeros(4, dtype=np.int32)
#     bb[0] = xmin * im_width   # left
#     bb[1] = xmax * im_width   # right
#     bb[2] = ymin * im_height  # bottom
#     bb[3] = ymax * im_height  # top
#     img_crop = cv2.rectangle(image, (bb[0], bb[2]), (bb[1], bb[3]), (0,255,0))
#     cv2.imshow('img_crp', img_crop)
# cv2.waitKey()



# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     # parser.add_argument('faceNet_model', type=str,
#     #                     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
#     parser.add_argument('image_path', type=str, help='Images to compare')
#
#
# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))