# MIT License
#
# Copyright (c) 2017 PXL University College
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

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import math
import time
sys.path.append("../src") # useful for the import of facenet in another folder
import facenet
import align.detect_face
from sklearn.cluster import DBSCAN


def main(model, data_dir_input, batch_size, image_size, margin, min_cluster_size, cluster_threshold, largest_cluster_only, gpu_memory_fraction):
    # pnet, rnet, onet = create_network_face_detection(gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
            start = time.time()
            facenet.load_model(model)
            loadmodel_time = time.time() - start
            print('load model cost #{}s'.format(loadmodel_time))

            # print(os.listdir(args.data_dir))
            for data_dir in os.listdir(data_dir_input):
                data_dir = os.path.join(data_dir_input, data_dir)
                nrof_images = len(os.listdir(data_dir))
                emb_array = np.zeros((nrof_images, 128))
                print('=========== folder dir (people) ===============')
                print('num of images in dir ', nrof_images)

                images_placeholder = sess.graph.get_tensor_by_name("input:0")
                embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

                nrof_batches = int(math.ceil(1.0*nrof_images/batch_size))
                for i in range(nrof_batches):
                    print('============== The #{} batch ============='.format(i))
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = os.listdir(data_dir)[start_index:end_index]
                    print('date_dir is ', data_dir)

                    image_batch = load_images_from_path_list(data_dir, paths_batch)
                    print('shape of image_batch ', np.shape(image_batch))
                    images = align_data(image_batch, image_size)#, margin, pnet, rnet, onet)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    if i < nrof_images-1:
                        emb_array[i*batch_size:(i+1)*batch_size, :] = sess.run(embeddings, feed_dict=feed_dict)
                    else:
                        emb_array[i*batch_size:-1, :] = sess.run(embeddings, feed_dict=feed_dict)


                print('num of faces: ', np.shape(emb_array)[0])
                matrix = np.zeros((np.shape(emb_array)[0], np.shape(emb_array)[0]))
                # print('========================')
                # print(emb_array)
                # print(type(emb_array))
                # print(np.shape(emb_array))
                # print('========================')
                # print(emb_array_tmp)
                # print(type(emb_array_tmp))
                # print(np.shape(emb_array_tmp))
                # print('')
                # # Print distance matrix
                # print('Distance matrix')
                # print('    ', end='')
                # for i in range(nrof_images):
                #     print('    %1d     ' % i, end='')
                # print('')
                for i in range(np.shape(emb_array)[0]):
                    for j in range(np.shape(emb_array)[0]):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb_array[i, :], emb_array[j, :]))))
                        matrix[i][j] = dist
                #         print('  %1.4f  ' % dist, end='')
                #     print('')
                #
                # print('')

                with open(data_dir + 'distance.txt', 'wb') as f:
                    np.savetxt(f, matrix, fmt='%.4f')

                # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
                db = DBSCAN(eps=cluster_threshold, min_samples=min_cluster_size, metric='precomputed')
                db.fit(matrix)
                labels = db.labels_

                # get number of clusters
                no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                print('No of clusters:', no_clusters)
                out_dir = data_dir + '/' + 'output'
                if no_clusters > 0:
                    if largest_cluster_only:
                        largest_cluster = 0
                        for i in range(no_clusters):
                            print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                            if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                                largest_cluster = i
                        print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
                        cnt = 1
                        test_la = np.nonzero(labels == largest_cluster)[0]
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                            for i in test_la:
                                image_path = os.path.join(data_dir, os.listdir(data_dir)[j])
                                ispng = image_path.endswith('.png')
                                if ispng:
                                    imag_re = misc.imread(image_path)
                                else:
                                    continue

                                misc.imsave(os.path.join(out_dir, str(cnt) + '.png'), imag_re)
                                cnt += 1
                    else:
                        print('Saving all clusters')
                        labels_ = sorted(list(labels), key=list(labels).count, reverse=True)
                        # labels = labels_
                        labels_set = set_not_by_sort(labels_)
                        for i in range(no_clusters):
                            cnt = 1
                            print('Cluster {}: {}'.format(i, np.nonzero(labels == labels_set[i])[0]))
                            path = os.path.join(out_dir, str(i))
                            if not os.path.exists(path):
                                os.makedirs(path)
                                for j in np.nonzero(labels == labels_set[i])[0]:
                                    image_path = os.path.join(data_dir, os.listdir(data_dir)[j])
                                    ispng = image_path.endswith('.png')
                                    if ispng:
                                        imag_re = misc.imread(image_path)
                                    else:
                                        continue

                                    misc.imsave(os.path.join(path, str(cnt) + '.png'), imag_re)
                                    cnt += 1
                            else:
                                for j in np.nonzero(labels == labels_set[i])[0]:
                                    imag_re = misc.imread(os.path.join(data_dir, os.listdir(data_dir)[j]))
                                    misc.imsave(os.path.join(path, str(cnt) + '.png'), imag_re)
                                    cnt += 1


def set_not_by_sort(x):
    re = [x[0]]
    for i in x:
        if re[-1] != i:
            re.append(i)
    return re



def align_data(image_list, image_size):#, margin, pnet, rnet, onet):
    img_list = []
    # minsize = 20  # minimum size of face
    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    # factor = 0.709  # scale factor
    # for x in range(len(image_list)):
    #     img_size = np.asarray(image_list[x].shape)[0:2]
    #     bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
    #     nrof_samples = len(bounding_boxes)
    #     if nrof_samples > 0:
    #         for i in range(nrof_samples):
    #             if bounding_boxes[i][4] > 0.95:
    #                 det = np.squeeze(bounding_boxes[i, 0:4])
    #                 bb = np.zeros(4, dtype=np.int32)
    #                 bb[0] = np.maximum(det[0] - margin / 2, 0)
    #                 bb[1] = np.maximum(det[1] - margin / 2, 0)
    #                 bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    #                 bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    #                 cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
    #                 aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    #                 prewhitened = facenet.prewhiten(aligned)
    #                 img_list.append(prewhitened)

    ### if the faces are detected and aligned by facenet,
    for x in range(np.shape(image_list)[0]):
        image_tmp = image_list[x]
        aligned = misc.imresize(image_tmp, (image_size, image_size), interp='bilinear')
        # print('image shape is ', np.shape(image_tmp))
        img_list.append(facenet.prewhiten(aligned))

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_path_list(folder, path_list):
    images = []
    for filename in path_list:
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == "__main__":

    model = '../2017'
    # data_dir_input = 'F:/raw_image_web_crop_multi'
    data_dir_input = 'F:/test_img'
    batch_size = 10
    image_size = 160
    margin = 44
    min_cluster_size = 1
    cluster_threshold = 0.4
    largest_cluster_only = False
    gpu_memory_fraction = 1.0

    main(model, data_dir_input, batch_size, image_size, margin, min_cluster_size, cluster_threshold, largest_cluster_only,
     gpu_memory_fraction)
