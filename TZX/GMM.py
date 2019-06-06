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
sys.path.append("src") # useful for the import of facenet in another folder

import facenet
import align.detect_face
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def main(args):
    n_classes = 50
    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    with tf.Graph().as_default():

        with tf.Session() as sess:
            facenet.load_model(args.model)

            image_list = load_images_from_folder(args.data_dir)
            images = align_data(image_list, args.image_size, args.margin, pnet, rnet, onet)

            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(images)
            print('embeddings', emb.shape)
            print('images', nrof_images)
            # matrix = np.transpose(emb)
            # matrix = emb
            # print('matrix',emb)

            matrix = np.zeros((nrof_images, nrof_images))

            print('')
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist
                    print('  %1.4f  ' % dist, end='')
                print('')

            print('')

            """
            matrix = np.zeros((nrof_images, nrof_images))
            for i in range(nrof_images):
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist
            """

            """


            #matrix = matrix[0:100]            
            n_components_range = range(100,150)
            bic = []
            aic = []
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components,covariance_type='full')
                gmm.fit(matrix)
                bic.append(gmm.bic(matrix))
                aic.append(gmm.aic(matrix))
            bic_min_index = bic.index(min(bic))
            aic_min_index = aic.index(min(aic))
            #print('min_index',min_index)
            plt.plot(n_components_range, bic)
            plt.plot(n_components_range, aic)
            plt.show()
            """
            """



            gmm = GaussianMixture(n_components= 2, covariance_type='diag')
            gmm.fit(matrix[0:5])
            score = gmm.score(matrix)
            pred_proba = gmm.predict_proba(matrix[6:9])
            params = gmm.get_params(deep=True)
            print('score',score)
            print('pred_proba', pred_proba)
            print('param',params)
            print('gmm means shape', gmm.means_.shape)
            print('gmm means',gmm.means_)
            """

            estimator = GaussianMixture(n_components=7, covariance_type='full')
            # estimator.fit(matrix[0:10])

            # estimator = BayesianGaussianMixture(n_components=2,weight_concentration_prior_type='dirichlet_distribution',
            # weight_concentration_prior=0.001,covariance_type='diag')
            # estimator = DBSCAN(eps=args.cluster_threshold, min_samples=args.min_cluster_size, metric='precomputed')
            estimator.fit(matrix)
            # estimator2 = BayesianGaussianMixture(n_components=3,max_iter=100,mean_precision_prior=1e-2,
            #    weight_concentration_prior=0.01,covariance_type='full')
            # estimator2.fit(matrix)
            pred = estimator.predict(matrix)
            proba = estimator.predict_proba(matrix[10:])
            scores = estimator.score_samples(matrix)
            print('pred', pred)
            print('proba', proba)
            print('scores', scores)
            print('n_components', estimator.n_components)
            print('means', estimator.means_)
            e_mean = estimator.means_
            # for i in range(e_mean.shape[0]):
            #    print('index',e_mean[i].index(min(e_mean[i])))
            print('shape of means', estimator.means_.shape)
            print('weights', estimator.weights_)
            # print(proba)

            # pred = estimator.predict_proba(matrix[15].reshape(1,-1))
            # pred = estimator.score(matrix[:,0].reshape(1,-1))

            no_clusters = len(set(pred)) - (1 if -1 in pred else 0)

            print('Saving all clusters')
            for i in range(no_clusters):
                cnt = 1
                # print('Cluster {}: {}'.format(i, np.nonzero(pred == i)[0]))
                print('Cluster {}: {}'.format(i, np.nonzero(pred == i)[0]))
                path = os.path.join(args.out_dir, str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
                    for j in np.nonzero(pred == i)[0]:
                        misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1
                else:
                    for j in np.nonzero(pred == i)[0]:
                        misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1

            """
            estimators = dict((cov_type, GaussianMixture(n_components=2,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

            for index, (name,estimator) in enumerate(estimators.items()):
                estimator.fit(matrix[0:])
                pred = estimator.predict(matrix)
                #pred = estimator.predict_proba(matrix[15].reshape(1,-1))
                #pred = estimator.score(matrix[:,0].reshape(1,-1))


                no_clusters = len(set(pred)) - (1 if -1 in pred else 0)

                print('Saving all clusters')
                for i in range(no_clusters):
                    cnt = 1
                    print('Cluster {}: {}'.format(i, np.nonzero(pred == i)[0]))
                    path = os.path.join(args.out_dir, str(i))
                    if not os.path.exists(path):
                        os.makedirs(path)
                        for j in np.nonzero(pred == i)[0]:
                            misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                            cnt += 1
                    else:
                        for j in np.nonzero(pred == i)[0]:
                            misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                            cnt += 1
            """

            """

            # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
            db = DBSCAN(eps=args.cluster_threshold, min_samples=args.min_cluster_size, metric='precomputed')
            db.fit(matrix)
            labels = db.labels_


            # get number of clusters
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            print('No of clusters:', no_clusters)

            if no_clusters > 0:
                if args.largest_cluster_only:
                    largest_cluster = 0
                    for i in range(no_clusters):
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                            largest_cluster = i
                    print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
                    cnt = 1
                    for i in np.nonzero(labels == largest_cluster)[0]:
                        misc.imsave(os.path.join(args.out_dir, str(cnt) + '.png'), images[i])
                        cnt += 1
                else:
                    print('Saving all clusters')
                    for i in range(no_clusters):
                        cnt = 1
                        print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                        path = os.path.join(args.out_dir, str(i))
                        if not os.path.exists(path):
                            os.makedirs(path)
                            for j in np.nonzero(labels == i)[0]:
                                misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                                cnt += 1
                        else:
                            for j in np.nonzero(labels == i)[0]:
                                misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                                cnt += 1
            """


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []

    for x in range(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)

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


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('data_dir', type=str,
                        help='The directory containing the images to cluster into folders.')
    parser.add_argument('out_dir', type=str,
                        help='The output directory where the image clusters will be saved.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=1)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--largest_cluster_only', action='store_true',
                        help='This argument will make that only the biggest cluster is saved.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
