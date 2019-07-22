from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
import math
from skimage import transform as trans

sys.path.append("../")  # useful for the import of facenet in another folder

import facenet
import align.detect_face
import random
from time import sleep

input("press Enter to continue")


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Load from model!')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        ####  alignment faces
                        # saver = tf.train.import_meta_graph('./mtcnn-3000000.meta')
                        # graph = tf.get_default_graph()
                        # sess.run(tf.global_variables_initializer())
                        # saver.restore(sess, './mtcnn-3000000')


                        bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet,
                                                                               threshold, factor)

                        if len(points) != 0:
                            nrof_faces = points.shape[1]

                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]

                                # ---------  show faces with box   -----
                                for i in range(nrof_faces):
                                    bb = det[i, :].astype(dtype=np.int32)
                                    img_and_crop = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
                                    cv2.putText(img, '# {}'.format(i), (bb[0], bb[3]-bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                                image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
                                cv2.imshow('img_crp', image_tmp)
                                cv2.waitKey()
                                ##########################

                                det_arr = []
                                _landmark = None
                                img_size = np.asarray(img.shape)[0:2]

                                if nrof_faces > 1:

                                    if args.detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            _landmark = points[:, i].reshape((2, 5)).T
                                            _bb = np.squeeze(bounding_boxes[i, 0:4])

                                            warp0, M = alignment(img, _landmark, img_size)
                                            ### crop aligned images
                                            bounding_boxes_new, point = align.detect_face.detect_face(warp0, minsize, pnet,
                                                                                                   rnet, onet,
                                                                                                  threshold, factor)

                                            if np.shape(bounding_boxes_new)[0] > 1:
                                                print(
                                                    '======find more faces after warp choose the most likely one! ======')
                                                offsets = [np.linalg.norm(dist(point[:, i])) for i in
                                                           range(np.shape(point)[1])]
                                                index_min = np.argmin(offsets)
                                                det = bounding_boxes_new[index_min, 0:4]
                                            elif np.shape(bounding_boxes_new)[0] == 0:
                                                print('\n')
                                                print('====== #{} have faces, while warp has none ======'.format(
                                                    output_filename))
                                                print('\n')
                                                det = _bb
                                            else:
                                                det = bounding_boxes_new[0, 0:4]

                                            # bb = det.astype(dtype=np.int32)
                                            # img_and_crop = cv2.rectangle(warp0, (bb[0], bb[1]), (bb[2], bb[3]),
                                            #                                  (0, 255, 0))
                                            # image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
                                            # cv2.imshow('img_crp', image_tmp)
                                            # cv2.waitKey()

                                            nrof_successfully_aligned = crop_face(i, det, args.margin, img_size, warp0,
                                                                                  args.image_size,
                                                                                  nrof_successfully_aligned,
                                                                                  args.detect_multiple_faces,
                                                                                  output_filename, text_file)

                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                        index = np.argmax(
                                            2*bounding_box_size - offset_dist_squared)  # some extra weight on the centering
                                        det_arr.append(det[index, :])
                                        _bbox = det[index, :]
                                        _landmark = points[:, index].reshape((2, 5)).T
                                        warp0, _ = alignment(img, _landmark, img_size)
                                        #### crop aligned images
                                        bounding_boxes_new, point = align.detect_face.detect_face(warp0, minsize, pnet, rnet,
                                                                                              onet,
                                                                                              threshold, factor)

                                        if np.shape(bounding_boxes_new)[0] > 1:
                                            print('======find more faces after warp choose the most likely one! ======')
                                            offsets = [np.linalg.norm(dist(point[:, i])) for i in range(np.shape(point)[1])]
                                            index_min = np.argmin(offsets)
                                            det = bounding_boxes_new[index_min, 0:4]
                                        elif np.shape(bounding_boxes_new)[0] == 0:
                                            print('\n')
                                            print('====== #{} have faces, while warp has none ======'.format(output_filename))
                                            print('\n')
                                            continue
                                        else:
                                            det = bounding_boxes_new[0, 0:4]


                                        nrof_successfully_aligned = crop_face(0, det, args.margin, img_size, warp0,
                                                                              args.image_size,
                                                                              nrof_successfully_aligned,
                                                                              False, output_filename, text_file)
                                else:
                                    _landmark = points.reshape((2, 5)).T
                                    warp0, M = alignment(img, _landmark, img_size)
                                    #### crop aligned images
                                    bounding_boxes_new, point = align.detect_face.detect_face(warp0, minsize, pnet, rnet, onet,
                                                                                          threshold, factor)

                                    if np.shape(bounding_boxes_new)[0] > 1:
                                        print('======find more faces after warp choose the most likely one! ======')
                                        offsets = [np.linalg.norm(dist(point[:, i])) for i in range(np.shape(point)[1])]
                                        index_min = np.argmin(offsets)
                                        det = bounding_boxes_new[index_min, 0:4]
                                    elif np.shape(bounding_boxes_new)[0] == 0:
                                        print('\n')
                                        print('====== #{} have faces, while warp has none ======'.format(output_filename))
                                        print('\n')
                                        continue
                                    else:
                                        det = bounding_boxes_new[0, 0:4]


                                    nrof_successfully_aligned = crop_face(0, det, args.margin, img_size, warp0,
                                                                          args.image_size, nrof_successfully_aligned,
                                                                          False, output_filename, text_file)

                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                os.remove(image_path)
                        else:
                            print('====  No faces in this image in {}, remove it! ===='.format(output_filename))
                            os.remove(image_path)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def visulize_tec(img, bb):
    img_and_crop = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0))
    image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
    cv2.imshow('img_crp', image_tmp)
    cv2.waitKey()


def crop_face(i, det, margin, img_size, img, image_size, nrof_successfully_aligned, detect_multiple_faces,
              output_filename, text_file):
    det = np.squeeze(det)
    bb = np.zeros(4, dtype=np.int32)
    if margin > 1:
        # =================================================
        # cropped with fixed margin which is used for lfw
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    else:
        # =================================================
        # cropped with percentage margin can be used for images download from internet
        width = det[2] - det[0]
        height = det[3] - det[1]
        bb[0] = np.maximum(det[0] - margin * width, 0)
        bb[1] = np.maximum(det[1] - margin * height, 0)
        bb[2] = np.minimum(det[2] + margin * width, img_size[1])
        bb[3] = np.minimum(det[3] + margin * height, img_size[0])

    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    nrof_successfully_aligned += 1
    filename_base, file_extension = os.path.splitext(output_filename)
    if detect_multiple_faces:
        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
    else:
        output_filename_n = "{}{}".format(filename_base, file_extension)
    misc.imsave(output_filename_n, scaled)
    text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
    return nrof_successfully_aligned


def dist(dst):
    _src = np.array([[30.2946, 51.6963], [65.5318, 51.5014],
                    [48.0252, 71.7366], [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)
    src_ = _src + 36.0
    src = src_.reshape((10,1)).T
    diff = dst - src
    return diff


def alignment(img, landmark, image_size):
    _src = np.array([[30.2946, 51.6963], [65.5318, 51.5014],
                    [48.0252, 71.7366], [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)
    src = _src + 36.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

    return warped, M


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=float,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_false')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
