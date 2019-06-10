from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from flask import Flask, request
import argparse

import scipy.stats as st
sys.path.append('serving')
from process import faceNet_serving_V0
from detect_face_np import load_and_align_data
import logging_flask
import code_message
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import  IOLoop
import tensorflow as tf
import json
import cv2 as cv
import numpy as np
import operator
import grpc


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

logger = logging_flask.get_logger(__name__)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('facenet', '192.168.1.254:9001', 'PredictionService host:port')

FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)
DATA = np.load('people_embs.npy').item()


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
        faces = load_and_align_data(img)
        if faces is None:
            print('No Faces in this image!')
            ret = {
                "file": f.filename,
                "code": 200,
                "message": "Has no faces in image, IMAGES PASS!",
                "result": "合规"

            }
        else:
            # print(faces.shape)
            emb = faceNet_serving_V0.img_to_emb_feature(faces, FACENET_CHANNEL)
            emb = list(emb)
            # print(len(emb))
            # print('shape of emb {} and ite type {}'.format(np.shape(emb), type(emb)))
            num_face = int(len(emb)/128)
            ret = {}
            maximum = []
            maximum_name = ['test']
            for i in range(num_face):
                emb_face = emb[i*128:(1+i)*128]
                likely = cal_sim(emb_face)
                ret[str(i)] = likely
                # print('=====================')
                maximum_name.append(max(likely, key=likely.get))

                maximum.append(likely[max(likely, key=likely.get)])
                # print('maximum {} and type {} '.format(maximum, type(maximum)))
                # print('maximum_name {} and type {} '.format(maximum_name, type(maximum_name)))
                # print('maximum {} and type {} '.format(max(maximum), type(max(maximum))))
                th = 0.10

                if max(maximum) < th:
                    ret = {
                        "file": f.filename,
                        "code": 300,
                        "message": "Has faces, IMAGES PASS!",
                        "result":  "合规"

                    }

                else:
                    index = list(np.where(np.array(maximum) > th)[0])
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
                            "data": data

                        }

    return json.dumps(ret)


def cal_sim(emb):

    # print('embedding vector: ',  emb)
    ###  load embedding features of critical peoples
    people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']

    emb_data = DATA

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

    emb_jack = emb
    print('emb_face shape', np.shape(emb_jack))
    jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
    likely = {}
    txt = {}
    for i in people:
        jack_dist[i]['dist_average'] = [
            np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
        jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
        jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean']) / (
        emb_data[i]['log_dist_std'])
        jack_dist[i]['Prob'] = st.norm.pdf(jack_dist[i]['Z_value']) / st.norm.pdf(0)
        # print('Prob is {}'.format(st.norm.pdf(jack_dist[i]['Z_value'])/st.norm.pdf(0)))
        # print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
        # print('The face is {:.2%} likely {}'.format(jack_dist[i]['Prob'][0], i))
        likely[i] = jack_dist[i]['Prob'][0]
    # LIKELY = []
    # for j in people:
    #     LIKELY.append(likely[i])
    return likely


if __name__ == '__main__':
    app.run('0.0.0.0',threaded = True)
    # threaded = true ,多线程，可保证检测不中断
    # http_server = HTTPServer(WSGIContainer(app))
    # http_server.listen(5000)
    # IOLoop.instance().start()



# def main(args):
#
#     images = detect(args.image_paths)
#     if images is None:
#         print('This image has no face inside')
#     else:
#         emb = [faceNet_serving_V0.img_to_emb_feature(images, "192.168.1.254:9001")]
#         cal_sim(emb)
#
# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('image_paths', type=str, help='File or dir')
#     return parser.parse_args(argv)
#
#
# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))

