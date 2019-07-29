from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from flask import Flask, request
import argparse

import scipy.stats as st
sys.path.append('serving')
# sys.path.append('../faceNet/src')
from process import faceNet_serving_V0
from detect_face_np import load_and_align_data
from logs import logging_flask
from code_message import code_message
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import  IOLoop
import tensorflow as tf
import json
import cv2
import numpy as np
import operator
import grpc


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

logger = logging_flask.get_logger(__name__)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('facenet', '192.168.1.254:9001', 'PredictionService host:port')


FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)
emb_data = np.load('people_embs_V1.npy').item()


@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # 301 file error
        cont = f.read()
        buf = np.frombuffer(cont, dtype=np.byte)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)


        faces, det_arr = load_and_align_data(img)
        print('file {} have #{} faces'.format(f, len(det_arr)))
        if faces is None:
            print('No Faces in this image!')
            ret = {
                "file": f.filename,
                "code": 200,
                "message": "Has_no_faces",
                "result": "合规"
            }
        else:
            det_arr_ser = np.reshape(det_arr, (1, np.size(det_arr)))
            emb = faceNet_serving_V0.img_to_emb_feature(faces, FACENET_CHANNEL)
            emb = list(emb)
            num_face = int(len(emb)/128)
            ret = {}
            maximum = []
            maximum_name = ['test']
            for i in range(num_face):
                emb_face = emb[i*128:(1+i)*128]
                likely = cal_sim_new(emb_face, emb_data)
                maximum_name.append(max(likely, key=likely.get))
                maximum.append(likely[max(likely, key=likely.get)])
                th = 0.40

                if max(maximum) < th:
                    ret = {
                        "file": f.filename,
                        "code": 300,
                        "message": "Has_face_pass",
                        "result":  "合规",
                        "det_arr": det_arr_ser.tolist()
                    }
                else:
                    index = list(np.where(np.array(maximum) > th)[0])
                    det_arr_tmp = np.array(det_arr)[index, :]
                    det_arr_ser = np.reshape(det_arr_tmp, (1, np.size(det_arr_tmp)))

                    data = []
                    for ii in index:
                        data.append(
                            {
                                "face_id": str(ii),
                                "user_name": maximum_name[ii+1],
                                "score": maximum[ii]
                            })

                    ret = {
                            "file": f.filename,
                            "code": 301,
                            "message": "敏感人物",
                            "result": "不合规",
                            "data": data,
                            "det_arr": det_arr_ser.tolist()
                    }
    return json.dumps(ret)


def cal_sim_new(emb, emb_data):
    likely = {}
    for i in emb_data:
        emb_feature = emb_data[i]['emb_ave']
        sim = feat_distance_cosine(emb, emb_feature)
        likely[i] = sim
    return likely


def cal_sim(emb, emb_data):
    people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
    attribute = ['dist_all', 'dist_average', 'dist_all_average']
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

    emb_jack = emb
    jack_dist = multi(people, attribute, {})
    likely = {}
    for i in people:
        jack_dist[i]['dist_average'] = [
            np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
        jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
        jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean']) / (
        emb_data[i]['log_dist_std'])
        jack_dist[i]['Prob'] = feat_distance_cosine(emb_jack, emb_data[i]['average_emb'])  # pro type is:  <class 'numpy.float64'>
        print('pro: ', jack_dist[i]['Prob'])
        print('pro type is: ', type(jack_dist[i]['Prob']))

        likely[i] = jack_dist[i]['Prob']
    return likely


def feat_distance_cosine(feat1, feat2):
    similarity = np.dot(feat1 / np.linalg.norm(feat1, 2), feat2 / np.linalg.norm(feat2, 2))
    return similarity


def feat_distance_l2(feat1, feat2):
    feat1_norm = feat1 / np.linalg.norm(feat1, 2)
    feat2_norm = feat2 / np.linalg.norm(feat2, 2)
    similarity = 1.0 - np.linalg.norm(feat1_norm - feat2_norm, 2) / 2.0
    return similarity


if __name__ == '__main__':
    app.run('0.0.0.0', threaded = True)



