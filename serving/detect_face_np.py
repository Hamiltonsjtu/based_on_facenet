import numpy as np
import tensorflow as tf
import os
from scipy import misc
import cv2
import sys
sys.path.append("../src") # useful for the import of facenet in another folder
import facenet
import align.detect_face


###  load image and detect faces in it, return faces

def load_and_align_data(image):

    ###  load input image
    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0
    detect_multiple_faces = True

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


    img_list = []

    bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    # print('============== boundary boxes =====================')
    # print(bounding_boxes)

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(
                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)              # bottom
            bb[1] = np.maximum(det[1] - margin / 2, 0)              # left
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])    # top
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])    # right
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            # print('cropped image shape: {}'.format(np.shape(scaled)))
            prewhitened = facenet.prewhiten(scaled)
            img_list.append(prewhitened)

        images = np.stack(img_list)
        print('This picture has {} faces'.format(nrof_faces))

    else:
        images = None
        # image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img_crp', image_tmp)
        # cv2.waitKey()
        # print('length of img_list is {}'.format(np.shape(img_list)))
        # print('append cropped images shape is {}'.format(np.shape(images)))

    return images