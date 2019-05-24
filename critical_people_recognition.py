"""
This is the main function used to do critical people recognition.
This function contains the following steps:
1. detect the faces for a given image
2. recognition based on the cropped faces
This project heavily based on FaceNet, which is develop by Google.
Too see more on Github: https://github.com/davidsandberg/facenet.git

Author: Shuai Zhu
Data: 20190523
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
from matplotlib import patches, lines
import matplotlib.pyplot as plt
import random
import colorsys
import copy
import argparse


sys.path.append("src/") # useful for the import of facenet in another folder
import facenet
import align.detect_face


def load_embedding_critical_people(emb_dir):
    """
    :param emb_critical_dir: the embedding vector direction for critical people
    :return: a dictionary with the names as key and vector as values
    """

    people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
    emb_critical_data = np.zeros((len(people), 512)) ## 512 present the dimension of embedding vector

    for i, name in enumerate(people):

        path = os.path.join(emb_dir, name + '_average_emb.txt')
        emb_tmp = np.loadtxt(path, delimiter=' ',  unpack=True) # 1d array of numpy can not be transposed.
        emb_critical_data[i,:] = emb_tmp

    return emb_critical_data


def detect_faces(args):
    """
    :param img_path: input image for face recognition
    :return: the bounding box and the cropped faces
    """

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    nrof_successfully_aligned = 0

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    try:
        img = misc.imread(args.img_dir[0])
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(args.img_dir, e)
        print(errorMessage)
    else:
        if img.ndim < 2:
            print('Unable to align "%s"' %args.img_dir)
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('number of faces is {}'.format(nrof_faces))
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
                    # print('type of det_arr {} and {}'.format(type(det_arr), det_arr))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        # print('det shape is {}'.format(np.shape(det_arr)))

        #####  save cropped faces in images
        scaled = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            crop_img = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
            # print(type(crop_img))

            prewhitened = facenet.prewhiten(crop_img)

            # print(np.shape(crop_img))
            scaled.append(prewhitened)
            nrof_successfully_aligned += 1
            # print('size of scaled faces is {}'.format(np.shape(scaled)))
    else:
        print('Unable to align "%s"' %args.img_dir)
    return img, det_arr, scaled


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visulize_img_crop(img_data, det_arr):
    # Display the image
    plt.imshow(img_data)
    # Get the current reference
    ax = plt.gca()
    # plt.figure()
    if np.shape(det_arr)[0] > 0:
        num_box = np.shape(det_arr)[0]
        # # Generate random colors
        colors = random_colors(num_box)
        #
        for i in range(num_box):
            color = colors[i]
            # Show area outside image boundaries.
            x1, y1, x2, y2 = det_arr[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

    plt.show()


def calculate_embedding(model_dir, scaled):
    """
    :param scaled:  calculate the embedding vector for a give scaled images
    :return: embedding vector
    """
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model_dir)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print('===============================================')
            print('feed image type {}'.format(type(scaled)))
            print('shpe of feed image {}'.format(np.shape(scaled)))

            feed_dict = {images_placeholder: scaled, phase_train_placeholder: False}
            ### Use the facenet model to calcualte embeddings
            # embedding_size = embeddings.get_shape()[1]
            # nrof_images = np.shape(scaled)[0]
            # emb_array = np.zeros((nrof_images, embedding_size))
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
    return emb_array


def calculate_distance_matrix(emb_array, emb_critical_data):
    """
    :param matrix: the input feature ebedding matrix, each row present the feature vector of one identity people's face,
                   and cols present the different faces
    :return: coorelation matrix
    """
    F = np.shape(emb_array)[0]
    P = np.shape(emb_critical_data)[0]
    dist = np.zeros((F,P))
    for i in range(F):
        for j in range(P):
            dist[i,j] = np.sqrt(np.sum(np.square(np.subtract(emb_array[i, :], emb_critical_data[j, :]))))
    return dist


def main(args):
    print('========== shuai start to work ==========')
    emb_standard = load_embedding_critical_people(args.emb_path)
    img_data, det_arr, scaled = detect_faces(args)
    emb_arr = calculate_embedding(args.model, scaled)
    print('============shape of emb==============')
    print(np.shape(emb_standard))
    print(emb_standard[0,:])
    print(np.shape(emb_arr))
    print(emb_arr[0,:])
    dist = calculate_distance_matrix(emb_arr, emb_standard)
    # print(dist)

    ### show the distance matrix if needed
    for i in range(np.shape(dist)[0]):
        print('     %1d     ' % i, end='')
    print('')
    for i in range(np.shape(dist)[0]):
        print('%1d  ' % i, end='')
        for j in range(np.shape(dist)[1]):
            print('  %1.4f  ' % dist[i,j], end='')
        print('')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('img_dir', type=str, nargs='+', help='direction of specific image')
    parser.add_argument('emb_path', type=str, help='directory containing the embedding vector with extension .txt')
    parser.add_argument('--batch_size', type=int, help='batch size for embedding', default=6)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

