from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import os
import cv2 as cv
import time
import shutil


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
import cv2
import scipy.stats as st


###  function to communicate with tensorflow_serving with help of grpc
def img_to_emb_feature(img, channel):
    print(img.shape)
    channel = grpc.insecure_channel(channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'facenet'
    request.model_spec.signature_name = 'calculate_embeddings'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, dtype=tf.float32))
    request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
    result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout
    result = result_tmp.outputs['embeddings'].float_val
    # request.model_spec.name = 'facenet'
    # request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    # request.inputs['input'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(img, shape=[1, img.shape[1], img.shape[2], img.shape[3]]))
    # result = stub.Predict(request, 10.0)  # 10 secs timeout
    # print("result: ", result)

    # boxes = np.array(result.outputs['embeddings'].float_val).reshape(
    #     result.outputs['detection_boxes'].tensor_shape.dim[0].size,
    #     result.outputs['detection_boxes'].tensor_shape.dim[1].size,
    #     result.outputs['detection_boxes'].tensor_shape.dim[2].size
    # )
    #
    # scores = np.array(result.outputs['detection_scores'].float_val)
    # detection_classes = np.array(result.outputs['detection_classes'].float_val)
    #
    # # num_detections = np.array(result.outputs['num_detections'].float_val)
    # boxes = np.squeeze(boxes)
    # scores = np.squeeze(scores)
    # height, width = img.shape[:2]
    #
    # pts_box = []
    # pts = None
    # door_img = None
    # scores_max = 0
    # detection_class = None
    # for i in range(boxes.shape[0]):
    #     if (scores[i] > 0.5) and (scores[i] > scores_max):
    #         scores_max = scores[i]
    #         ymin, xmin, ymax, xmax = boxes[i]
    #         ymin = int(ymin * height)
    #         ymax = int(ymax * height)
    #         xmin = int(xmin * width)
    #         xmax = int(xmax * width)
    #
    #         pts = np.array([xmin, ymin, xmax, ymax])
    #
    #         detection_class = detection_classes[i]
    #
    # # channel.close()
    return result

###  load image and detect faces in it, return faces
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
    if os.path.isdir(image_paths):
        tmp_image_paths = os.listdir(image_paths)
    else:
        tmp_image_paths = [image_paths]

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
            # image_tmp = cv2.cvtColor(img_and_crop, cv2.COLOR_BGR2RGB)
            # cv2.imshow('img_crp', image_tmp)
            # cv2.waitKey()
            # print('length of img_list is {}'.format(np.shape(img_list)))
            # print('append cropped images shape is {}'.format(np.shape(images)))
    return images

###  load input image
image_path = './hu.jpg'
image_size = 160
margin = 44
gpu_memory_fraction = 1.0
detect_multiple_faces = True

print(image_path)
images = load_and_align_data(image_path, image_size, margin, gpu_memory_fraction,
                             detect_multiple_faces)


###  sent cropped face to faceNet model which is disployed by tensorflow_serving
emb = [img_to_emb_feature(images, "192.168.1.254:9001")]
num_face = int(np.shape(emb)[1]/128)
emb = np.asarray(emb).reshape((num_face, 128))
# print('embedding vector: ',  emb)
###  load embedding features of critical peoples
people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
emb_data = np.load('people_embs.npy').item()


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


print('type of emb {} and its shape: {}'.format(type(emb), np.shape(emb)))

for l in range(np.shape(emb)[0]):
    print('======== START CALCULATE THE {}-th face DISTANCE ========'.format(l))
    emb_jack = emb[l, :]
    print('emb_face shape', np.shape(emb_jack))
    jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
    for i in people:
        jack_dist[i]['dist_average'] = [
            np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
        jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
        jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean']) / (
        emb_data[i]['log_dist_std'])
        jack_dist[i]['Prob'] = st.norm.pdf(jack_dist[i]['Z_value']) / st.norm.pdf(0)
        # print('Prob is {}'.format(st.norm.pdf(jack_dist[i]['Z_value'])/st.norm.pdf(0)))
        # print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
        print('The face is {:.2%} likely {}'.format(jack_dist[i]['Prob'][0], i))



