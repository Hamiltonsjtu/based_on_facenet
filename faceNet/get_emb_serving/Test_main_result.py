from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from flask import Flask, request
from process import faceNet_serving_V0
from detect_face_np import load_and_align_data

import tensorflow as tf
import json
import cv2
import numpy as np
import operator
import grpc


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('facenet', '192.168.1.254:9001', 'PredictionService host:port')
FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)


class_names = np.load('class.npy')
file_name = np.load('name.npy')
embs = np.load('embs.npy')

cls_names = list(set(class_names))
data = {}
data_ave = {}
for i in cls_names:
    indice = np.where(class_names == i)[0]
    data[i] = embs[indice, :]
    data_ave[i] = np.mean(data[i], axis=0)
print('data ave size', np.shape(data_ave))

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
        print('image size', np.shape(img))
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
            print('input image shape ', np.shape(faces))

            emb = faceNet_serving_V0.img_to_emb_feature(faces, FACENET_CHANNEL)
            print('emb size', len(emb))
            emb = list(emb)
            num_face = int(len(emb)/512)
            ret = {}
            maximum = []
            maximum_name = ['test']
            for i in range(num_face):
                emb_face = emb[i*512: (1+i)*512]
                likely = cal_sim_new(emb_face, data_ave)
                print('Likely ', likely)
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
        emb_feature = emb_data[i]
        sim = feat_distance_cosine(emb, emb_feature)
        likely[i] = sim
    return likely


def cal_sim(emb, emb_data):
    people = ['xijinping_baidu', 'hujintao_baidu', 'jiangzemin_baidu', 'dengxiaoping_baidu', 'wenjiabao_baidu', 'maozedong_baidu', 'zhouenlai_baidu']
    emb_jack = emb
    likely = {}
    for i in people:
        likely[i] = feat_distance_cosine(emb_jack, emb_data[i])  # pro type is:  <class 'numpy.float64'>
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



