from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("src")                  # FACENET
sys.path.append('SSD_face_detection')   # SSD
from SSD_face_detection.SSD_detect_face import ssd_find_face as SSD
import facenet
import align.detect_face
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
img_path = 'xi_tram_ce.jpg'  #args.image_path


##### =======  detect face for image  ====== #######
boxes, scores, face_indice, im_height, im_width, image = SSD(img_path)
# image = cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB)
det_arry = boxes[face_indice, :]
scores_arr = scores[face_indice]
# print('face index {}'.format(face_indice))
# print('shape of image {}'.format([im_height, im_width]))
# print('det_arry: {}'.format(det_arry))
# print('scores: {}'.format(scores_arr))


##### =======  landmark and project for faces  ====== #######
predictor_model = "dlib/shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = open_Face.AlignDlib(predictor_model)

# win = dlib.image_window()
# win.set_image(image)

for i in face_indice:
    box = boxes[i, :]
    xmin = box[1]
    xmax = box[0]
    ymin = box[3]
    ymax = box[2]

    print('xmin {}, ymin {}, xmax {}, ymax {}'.format(xmin, ymin, xmax, ymax))
    # img_crop = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0))
#     cv2.imshow('img_crp', img_crop)
# cv2.waitKey()

    det_arry_dlib = dlib.rectangle(xmin, ymin, xmax, ymax)
    print(det_arry_dlib)
    # Draw a box around each face we found
    # win.add_overlay(det_arry_dlib)
    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, det_arry_dlib)
    # print('=========================')
    # print(pose_landmarks)
    # Use openface to calculate and perform the face alignment

    alignedFace = face_aligner.align(534, image, det_arry_dlib, landmarkIndices=open_Face.AlignDlib.OUTER_EYES_AND_NOSE)
    # cv2.imwrite("aligned_face_jack{}.jpg".format(i), alignedFace)

    # win.add_overlay(pose_landmarks)
# dlib.hit_enter_to_continue()


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