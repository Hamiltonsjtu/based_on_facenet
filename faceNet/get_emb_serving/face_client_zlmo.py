# ! /usr/bin/env python
# coding=utf-8

import time
import numpy as np
import tensorflow as tf
import grpc
import requests
import time
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
#
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('facenet', '192.168.1.252:8500', 'PredictionService host:port')
# FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)

image = cv2.imread(r'F:\SSD_baidu_crop\dengxiaoping_cropped\dengxiaoping_0000_0.png')
image1 = np.array(image)
# batch_size = 10
# image1 = np.stack([image1] * batch_size) #for batch test
image = np.expand_dims(image, axis=0)
print('image size', image.shape)

channel = grpc.insecure_channel('192.168.1.254:9001')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# Send request
# See prediction_service.proto for gRPC request/response details.
request = predict_pb2.PredictRequest()


request.model_spec.name = 'facenet'
request.model_spec.signature_name = 'calculate_embeddings'
# print('request ', request)
request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image, shape=[image.shape[0], image.shape[1],image.shape[2], 3],dtype=tf.float32))
request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
print('========================')
result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout

# results = stub.Predict(request, 10.0)
print(result_tmp)
