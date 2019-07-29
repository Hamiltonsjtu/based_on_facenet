# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# !/usr/bin/env python2.7
from __future__ import print_function

# This is a placeholder for a Google-internal import.

import cv2
import grpc
import requests
import tensorflow as tf
from PIL import Image
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', '192.168.1.254:9001',
                           'PredictionService host:port')
# tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):

    image = cv2.imread(r'F:\SSD_cropped\xucaihou\xucaihou_0000.jpg')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    print('image size', image.shape)

    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'facenet'
    request.model_spec.signature_name = 'calculate_embeddings'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image,shape=[image.shape[0], image.shape[1],image.shape[2], 3], dtype=tf.float32))
    request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    # print(size(result))
    response = np.array(result.outputs['embeddings'].float_val)
    response = response.reshape(-1, 128)
    print(response)
    print(response.shape)
    channel.close()


if __name__ == '__main__':
    tf.app.run()

