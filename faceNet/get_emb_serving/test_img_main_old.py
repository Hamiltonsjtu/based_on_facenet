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

emb = np.load('./20190627_emb_feature_small_3_160/embeddings.npy')
labels_str = np.load('./20190627_emb_feature_small_3_160/label_strings.npy')
labels_num = np.load('./20190627_emb_feature_small_3_160/labels.npy')

peoples = list(set(labels_str))

emb_dict = {}

for i in peoples:
    index = np.where(labels_str == i)[0]
    emb_ = emb[index, :]
    emb_ave = np.mean(emb_, axis = 0)
    emb_dict[i] = {'emb': emb_, 'emb_ave': emb_ave}

# cls_names = list(set(class_names))
# data = {}
# data_ave = {}
# for i in cls_names:
#     indice = np.where(class_names == i)[0]
#     data[i] = embs[indice, :]
#     data_ave[i] = np.mean(data[i], axis=0)
# print(data)

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

        faces, det_arr, _ = load_and_align_data(img)
        print('file {} have #{} faces'.format(f, len(det_arr)))
        if faces is None:
            print('No Faces in this image!')
            ret = {
                "file": f.filename,
                "code": 200,
                "message": "Has_no_faces",
                "result": "合规"
                # "img": {"size": list(np.shape(img)),"0": img_0_ser, "1": img_1_ser, "2": img_2_ser}
            }
        else:
            det_arr_ser = np.reshape(det_arr, (1, np.size(det_arr)))
            print('input face shape ', np.shape(faces))

            emb = faceNet_serving_V0.img_to_emb_feature(faces, FACENET_CHANNEL)

            print('emb size', len(emb))
            emb = list(emb)
            num_face = int(len(emb)/128)
            ret = {}
            maximum = []
            maximum_name = ['test']
            for i in range(num_face):
                emb_face = emb[i*128: (1+i)*128]
                likely = cal_sim_new(emb_face, emb_dict)
                print('Likely ', likely)
                # print('Likely_l2', likely_l2 )
                maximum_name.append(max(likely, key=likely.get))
                maximum.append(likely[max(likely, key=likely.get)])
                th = 0.80

                if max(maximum) < th:
                    ret = {
                        "file": f.filename,
                        "code": 300,
                        "message": "Has_face_pass",
                        "result":  "合规",
                        "det_arr": det_arr_ser.tolist()
                        # "img":  {"size": list(np.shape(img)), "0": img_0_ser, "1": img_1_ser, "2": img_2_ser}
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
                            # "img":  {"size": list(np.shape(img)), "0": img_0_ser, "1": img_1_ser, "2": img_2_ser},
                            "det_arr": det_arr_ser.tolist()
                    }
    return json.dumps(ret)


def cal_sim_new(emb, emb_data):
    likely = {}
    for i in list(emb_data.keys()):
        emb_feature = emb_data[i]['emb_ave']
        likely[i] =  feat_distance_cosine(emb, emb_feature)
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



