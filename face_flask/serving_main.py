#coding:utf-8
from flask import Flask, request
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import  IOLoop
import send_predict
from process import faceNet_client
from process import ssd_face_detect_client
from process import face_util
from process import parse_result
import scipy.stats as st

import grpc
import json
import cv2 as cv
import numpy as np

from logger import code_message
from logger import flask_logger
logger = flask_logger.get_logger(__name__)

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


FACENET_CHANNEL = grpc.insecure_channel('192.168.1.254:9001')
FACE_CHANNEL = grpc.insecure_channel('192.168.1.254:9002')


# DATA = np.load('people_embs.npy', allow_pickle=True).item()
emb = np.load('./20190627_emb_feature_small_3_160/embeddings.npy')
labels_str = np.load('./20190627_emb_feature_small_3_160/label_strings.npy')
labels_num = np.load('./20190627_emb_feature_small_3_160/labels.npy')
DATA = {}

peoples = list(set(labels_str))
for i in peoples:
    index = list(np.where(labels_str == i)[0])
    DATA[i] = np.mean(emb[index, :], axis=0)

@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/v1/face_censor', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']  # 301 file error
            cont = f.read()
            buf = np.frombuffer(cont, dtype=np.byte)
            img = cv.imdecode(buf, cv.IMREAD_COLOR)
        except Exception as e:
            logger.error(e)
            return json.dumps(
                {'filename': f.filename, 
                 "code": code_message.invalid_image_code, 
                 "message": str(e)})
        _result_dict = send_predict.get_faces_and_predict(img, FACE_CHANNEL, FACENET_CHANNEL, DATA, f.filename)
        print(_result_dict)
    return json.dumps(_result_dict)


if __name__ == '__main__':
    #app.run('0.0.0.0', threaded = True)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5002)
    IOLoop.instance().start()

