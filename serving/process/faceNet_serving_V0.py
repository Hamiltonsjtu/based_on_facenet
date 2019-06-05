
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

from scipy import misc
import tensorflow as tf
import numpy as np
import time
import os
import copy
import argparse
import cv2
import scipy.stats as st


###  function to communicate with tensorflow_serving with help of grpc
def img_to_emb_feature(img, channel):
    # print(img.shape)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'facenet'
    request.model_spec.signature_name = 'calculate_embeddings'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, dtype=tf.float32))
    request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
    result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout
    # print(result_tmp)
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
