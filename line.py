"""
This script calculate the embedding vector for one picture.
Picture may contain several faces, so this script contains these steps:
1. detect face
2. crop face
3. calculate the embedding vector for corpped faces
Reture embedding vector [n, 128]
n presents number of faces
Author: Shuai Zhu @TZX
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("src")  # useful for the import of facenet in another folder

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


def main(args):

    #### load emb data
    people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
    # attrib = ['emb', 'distance', 'average_emb']
    # emb_data = load_emb(people, attrib)
    # np.save('people_embs.npy', emb_data)
    ####
    emb_data = np.load('people_embs.npy').item()
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction, args.detect_multiple_faces)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            start_time = time.time()
            facenet.load_model(args.model)
            end_time = time.time()
            print('loading model costs {}s'.format(end_time - start_time))
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            print('shape of embedding is {}'.format(np.shape(emb)))
            print('==============================================')

    n = np.shape(emb)[0]
    for l in range(n):
        print('======== START CALCULATE THE {}-th face DISTANCE ========'.format(l))
        emb_jack = emb[l, :]
        jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
        for i in people:
            jack_dist[i]['dist_average'] = [
                np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
            jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
            jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean'])/(emb_data[i]['log_dist_std'])
            jack_dist[i]['Prob'] = st.norm.pdf(jack_dist[i]['Z_value'])/st.norm.pdf(0)
            # print('Prob is {}'.format(st.norm.pdf(jack_dist[i]['Z_value'])/st.norm.pdf(0)))
            # print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
            print('The face is {:.2%} likely {}'.format(jack_dist[i]['Prob'][0], i))


def load_emb(people, attrib):
    emb_data = multi(people, attrib, {})
    for i in people:
        for j in attrib:
            emb_data[i][j] = np.loadtxt('data/images_cropped/' + i + '/' + j + '.txt', delimiter=' ')
        # print('num of standard images for {} is {}'.format(i, np.shape(emb_data[i]['emb'])))
        emb_data[i]['dist_emb'] = [np.sqrt(np.sum(np.square(np.subtract(k, emb_data[i]['average_emb'])))) for k in
                                   emb_data[i]['emb']]
        emb_data[i]['log_dist'] = np.log(emb_data[i]['dist_emb'])
        emb_data[i]['log_dist_mean'] = np.mean(emb_data[i]['log_dist'])
        emb_data[i]['log_dist_std'] = np.std(emb_data[i]['log_dist'])
        print('distance shape for [ {} ] is {}'.format(i, np.shape(emb_data[i]['distance'])))
    return emb_data


def multi(*args):
    """
    Build multiple level dictionary for python
    For example:
        multi(['a', 'b'], ['A', 'B'], ['1', '2'], {})
    returns
        {   'a': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}},
            'b': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}}}
    """
    if len(args) > 1:
        return {arg:multi(*args[1:]) for arg in args[0]}
    else:
        return args[0]


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction,detect_multiple_faces):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        print(img.shape)
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        print('============== boundary boxes =====================')
        print(bounding_boxes)

        nrof_faces = bounding_boxes.shape[0]
        print('This picture has {} faces'.format(nrof_faces))
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
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
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                print('cropped image shape: {}'.format(np.shape(scaled)))
                prewhitened = facenet.prewhiten(scaled)
                img_list.append(prewhitened)
                img_and_crop = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
            images = np.stack(img_list)
            image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
            cv2.imshow('img_crp', image_tmp)
            cv2.waitKey()
            print('length of img_list is {}'.format(np.shape(img_list)))
            print('append cropped images shape is {}'.format(np.shape(images)))
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))