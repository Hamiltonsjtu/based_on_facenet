from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from flask import Flask, request
import json
import cv2
import numpy as np
import operator
import grpc
import pandas as pd
from process import faceNet_serving_V0


# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


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

        # print('buf {} and buf size {}'.format(buf, np.shape(buf)))
        img = np.expand_dims(cv2.imdecode(buf, cv2.IMREAD_COLOR), axis=0)
        print('input image shape ', np.shape(img))
        emb = faceNet_serving_V0.img_to_emb_feature(img, FACENET_CHANNEL)

        print('emb size ', len(emb))
        ret = {
            "file": f.filename,
            "emb": list(emb)
        }
        print('return: ', ret)

    return json.dumps(ret)


if __name__ == '__main__':
    app.run('0.0.0.0', threaded = True)



