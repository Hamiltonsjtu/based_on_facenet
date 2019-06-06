"""
Performs face alignment and calculates L2 distance between the embeddings of images.
Calculate the distance of pictures for a given folder
"""

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
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from sklearn.cluster import KMeans
sys.path.append("src") # useful for the import of facenet in another folder
import src.facenet as facenet
import align.detect_face


def main(args):
    image_path_tmp = os.listdir(args.image_path[0])
    # print(image_path_tmp)
    image_paths = [args.image_path[0] + '/' + f for f in image_path_tmp]
    # print(image_paths)
    # image_files = load_and_align_data(image_paths, args.image_size, args.margin, args.gpu_memory_fraction)
    all_image_path_num = len(image_path_tmp)
    chosen_img_index = get_index_every_nth(image_paths, args.batch_size)
    print(chosen_img_index)
    print('===============++++++++++++++++++++================')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            emb = np.zeros((all_image_path_num, 128))
            # Run forward pass to calculate embeddings
            for k in np.arange(len(chosen_img_index)-1):
                # print('=====================k is {}=============='.format(k))
                image_sub_files = image_paths[chosen_img_index[k]+1:chosen_img_index[k+1]+1]
                print('sub files path {}'.format(image_sub_files))
                images = load_images_from_folder(image_sub_files, args.image_size)
                print('images shape {}'.format(np.shape(images)))
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb[chosen_img_index[k]+1:chosen_img_index[k+1]+1, :] = sess.run(embeddings, feed_dict=feed_dict)
                # print('num of images is {} and num of embedding is {}'.format(np.shape(image_sub_files), np.shape(embeddings)))

    dist = calculate_distance_matrix(emb)
    folder_name = args.out_put_emb[0]
    print(dist)
    print('=========================')
    print(folder_name)
    print('========  distance average =============')
    print(np.sum(dist)/np.size(dist))
    with open(folder_name + '/' + 'emb.txt', 'wb') as f:
        np.savetxt(f, emb, fmt='%1.6f')
    with open(folder_name + '/' + 'average_emb.txt', 'wb') as f:
        np.savetxt(f, np.mean(emb, axis=0), fmt='%1.6f')
    with open(folder_name + '/' + 'distance.txt', 'wb') as f:
        np.savetxt(f, dist, fmt='%.4f')


def load_images_from_folder(paths, image_size):
    image_list = []
    for filename in paths:
        img = misc.imread(filename)
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')
        # print('========================')
        # print(img)
        # print('img shape {}'.format(np.shape(img)))
        prewhitened = facenet.prewhiten(img)
        image_list.append(prewhitened)
        print('shape of image_list {}'.format(np.shape(image_list)))

        images = np.stack(image_list)
    return images


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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_path', type=str, nargs='+', help='Folder')
    parser.add_argument('out_put_emb', type=str, nargs='+', help='Folder')
    parser.add_argument('--batch_size', type=int, help='batch size for embedding', default=20)
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
