"""
Performs face alignment and calculates L2 distance between the embeddings of images.
Calculate the distance of pictures for a given folder
"""


# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import copy
import argparse
from sklearn.cluster import KMeans
import src.facenet as facenet
import src.align.detect_face as detect_face


def main(args):
    image_files = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    all_image_path_num = len(image_files)
    chosen_img_index = get_index_every_nth(image_files, args.batch_size)
    print(chosen_img_index)
    print('===============++++++++++++++++++++================')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            emb = np.zeros((all_image_path_num, 512))
            # Run forward pass to calculate embeddings
            for k in np.arange(len(chosen_img_index)-1):
                # print('=====================k is {}=============='.format(k))
                image_sub_files = image_files[chosen_img_index[k]+1:chosen_img_index[k+1]+1]
                # print('=========== image sub files ===============')
                # print(image_sub_files)
                # print(image_files)
                images = load_png_files(image_sub_files,args.image_size)
                print('===============================================')
                print('feed image type {}'.format(type(images)))
                print('shpe of feed image {}'.format(np.shape(images)))
                # images = load_and_align_data(image_sub_files, args.image_size, args.margin, args.gpu_memory_fraction)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                # print(np.shape(emb))
                # print(sess.run(embeddings, feed_dict=feed_dict))
                # print('============+++++++++++++++++=================')
                # print('chosen image index from {} to {}'.format(chosen_img_index[k]+1,chosen_img_index[k+1]+1))
                # print('embedding size is {}'.format(sess.run(embeddings,feed_dict=feed_dict)))
                emb[chosen_img_index[k]+1:chosen_img_index[k+1]+1, :] = sess.run(embeddings, feed_dict=feed_dict)


    dist = calculate_distance_matrix(emb)
    folder_name = args.folder_path[0]

    with open(os.path.join(folder_name, 'emb.txt'), 'wb') as f:
        np.savetxt(f, emb, fmt='%1.6f')
    with open(os.path.join(folder_name, 'average_emb.txt'), 'wb') as f:
        np.savetxt(f, np.mean(emb, axis=0), fmt='%1.6f')
    with open(os.path.join(folder_name, 'distance.txt'), 'wb') as f:
        np.savetxt(f, dist, fmt='%.4f')

    ### show the distance matrix if needed
    for i in range(all_image_path_num):
        print('%1d: %s' % (i, image_files[i]))
    print('')
    for i in range(np.shape(dist)[0]):
        print('     %1d     ' % i, end='')
    print('')
    for i in range(np.shape(dist)[0]):
        print('%1d  ' % i, end='')
        for j in range(np.shape(dist)[1]):
            print('  %1.4f  ' % dist[i,j], end='')
        print('')


# def cluster_not_same_class_image(embedding_feature):
#     """
#     :param embedding_matrix: the embedding feature matrix
#     :return: True or False
#     """
#     kmeans = KMeans(n_clusters=2, random_state=0).fix(embedding_feature)

            # print('feature embedding shape is {}'.format(np.shape(emb)))
            #
            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, image_files[i]))
            # print('')
            #
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            # for i in range(nrof_images):
            #     print('%1d  ' % i, end='')
            #     for j in range(nrof_images):
            #         dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            #         print('  %1.4f  ' % dist, end='')
            #     print('')


def calculate_distance_matrix(matrix):
    """
    :param matrix: the input feature ebedding matrix, each row present the feature vector of one identity people's face,
                   and cols present the different faces
    :return: coorelation matrix
    """
    n, m = np.shape(matrix)
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(np.sum(np.square(np.subtract(matrix[i, :], matrix[j, :]))))

    return dist


def get_index_every_nth(given_list, n):
    """
    :param given_list:
    :param n: every nth element gap
    :return: sub_index, gap boundary index
    for example:
    a = [4 2 3 9 6 6 8 5]
    b = get_index_every_nth(a, 3)
    :return
    c = [0,2,3,5,6,7]
    """
    num_list = len(given_list)
    mod = divmod(num_list, n)
    diviser = mod[0]
    res = mod[1]
    re = [-1]
    for i in np.arange(diviser):
        i += 1
        re.append(i*n-1)
    if res != 0:
        re.append(num_list-1)
    return re



def load_png_image_of_folder(folder_path):
    image_paths = []
    folder_path = folder_path[0]
    # print(folder_path)
    # current_dir = os.getcwd()
    # folderpath = os.path.join(current_dir,folder_path)
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            image_paths.append(os.path.join(folder_path, file))
    return image_paths


def load_png_files(image_paths, image_size):
    """Due to we have cropped the images for faces, there is no need to redo MTCNN net"""
    img_list = []
    for image in image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')
        img_list.append(img)

    images = np.stack(img_list)
    return images


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        # print('======================')
        # print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            img_list.append(img)
            # image_paths.remove(image)
            print("can't detect face, do not crop this one ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        # print('====================')
        # print(np.shape(img_list))
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Folder')
    parser.add_argument('--batch_size', type=int, help='batch size for embedding', default=6)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--is_aligned', type=str,
        help='Is the data directory already aligned and cropped?', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
