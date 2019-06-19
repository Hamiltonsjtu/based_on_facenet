from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request

# from logs import logging_flask
# from code_message import code_message

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import tensorflow as tf
import json
import cv2 as cv
import numpy as np
import grpc


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# logger = logging_flask.get_logger(__name__)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('facenet', '192.168.1.254:9001', 'PredictionService host:port')

FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)


@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # 301 file error
        cont = f.read()
        buf = np.frombuffer(cont, dtype=np.byte)
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
        faces = load_face(img)
        emb = img_to_emb_feature(faces, FACENET_CHANNEL)
        print(type(emb))
        # print()
        ret = {
                "file": f.filename,
                "code": 200,
                "emb": list(emb)
        }
        print(ret)
    return json.dumps(ret)


def load_face(image):
    img_list = []
    prewhitened = prewhiten(image)
    img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


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
    result = result_tmp.outputs['embeddings'].float_val

    return result

if __name__ == '__main__':
    app.run('0.0.0.0',threaded = True)



