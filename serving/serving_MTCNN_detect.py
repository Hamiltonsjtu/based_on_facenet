from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request

import tensorflow as tf
import json
import cv2 as cv
import numpy as np
import grpc

from process import MTCNN_detect_face

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# logger = logging_flask.get_logger(__name__)
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('facenet', '192.168.1.254:9001', 'PredictionService host:port')
tf.app.flags.DEFINE_string('mtcnn', '192.168.1.254:9900', 'PredictionService host:port')

MTCNN_CHANNEL = grpc.insecure_channel(FLAGS.mtcnn)

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

        ret, points = MTCNN_detect_face.detect_face(img)

    return json.dumps(ret)



if __name__ == '__main__':
    app.run('0.0.0.0',threaded = True)






