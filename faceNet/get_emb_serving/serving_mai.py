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


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
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
        img = np.expand_dims(cv2.imdecode(buf, cv2.IMREAD_COLOR), axis=0)
        # print(f)
        # print(cont)
        # print(buf)
        # print("size of img", np.shape(img))
        # det_arr_ser = np.reshape(img, (1, np.product(img.shape)))[0]
        # img_ser = pd.Series(det_arr_ser).to_json(orient='values')
        # print("size of img_ser", np.shape(img_ser))
        emb = img_to_emb_feature(img, FACENET_CHANNEL)

        ret = {
            "file": f.filename,
            # "image_data": img_ser,
            "emb": list(emb)
        }
        print('return: ', ret)

    return json.dumps(ret)


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

    return result


if __name__ == '__main__':
    app.run('0.0.0.0', threaded = True)



