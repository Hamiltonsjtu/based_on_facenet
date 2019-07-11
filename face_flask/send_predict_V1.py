#coding:utf-8
from process import faceNet_client
from process import ssd_face_detect_client
from process import face_util
from process import parse_result
import scipy.stats as st
import numpy as np
import time
from logger import code_message
from logger import flask_logger
logger = flask_logger.get_logger(__name__)


def get_faces_and_predict(img, face_channel, facenet_channel, emb_dict, filename):
    face_dict = ssd_face_detect_client.get_face_box_update(img, face_channel)

    if face_dict['code'] == code_message.no_face_detect_code:
        ret = {"file": filename,
            "code": code_message.no_face_detect_code,
            "message": "Has_no_faces",
            "result": "合规"}

    else:
        faces, det_arr = face_util.crop_ssd_face_img_update(img, face_dict['data'], filename)
        print('det_arr ', det_arr)
        det_arr_ser = np.reshape(det_arr, (1, np.size(det_arr)))
        print('faces shape', np.shape(faces))
        faceNet_dict = faceNet_client.img_to_emb_feature_update(faces, facenet_channel)
        emb = faceNet_dict['data']
        emb = list(emb)
        num_face = int(len(emb)/128)
        ret_not_pass = {}
        ret_pass = {}
        data = []
        for i in range(num_face):
            emb_face = emb[i*128: (1+i)*128]
            likely = cal_sim_new(emb_face, emb_dict)
            theta = {'xijinping': [0.75],
                     'hujintao': [0.73],
                     'jiangzemin': [0.72],
                     'dengxiaoping': [0.60],
                     'wenjiabao': [0.71],
                     'maozedong': [0.85],
                     'zhouenlai': [0.78]}
            diff_name = list([i for i, _ in theta.items()])
            value = []
            for name in diff_name:
                theta[name].append(likely[name])
                value.append(theta[name])
            value = np.squeeze(value)
            # print('dictionary theta', theta)
            Flag_all = listsbigger(value[:,0], value[:,1])
            if all(Flag_all):
                value_max = np.max(value[:,1])
                value_max_indice = np.argmax(value[:,1])
                value_max_name = diff_name[value_max_indice]
                ret_pass = {
                    "file": filename,
                    "code": 300,
                    "message": "Has_face_pass",
                    "result":  "合规",
                    "user_name": str(value_max_name),
                    "score": str(value_max),
                    "det_arr": det_arr_ser.tolist()
                }
            else:
                index_False = [i for i, x in enumerate(Flag_all) if not x]
                max_term_tmp = value[index_False, :]
                delta_score = max_term_tmp[:, 1] - max_term_tmp[:, 0]
                delta_score_max_indice = np.argmax(delta_score)
                max_term = max_term_tmp[delta_score_max_indice, 1]
                max_name_tmp = [diff_name[i] for i in index_False]
                max_name = max_name_tmp[delta_score_max_indice]
                # print('max_name {} and its value {}'.format(max_name, max_term))
                det_arr_tmp = np.array(det_arr)[i, :]
                det_arr_ser = np.reshape(det_arr_tmp, (1, np.size(det_arr_tmp)))

                data.append({
                        "face_id": str(i),
                        "user_name": str(max_name),
                        "score": str(max_term),
                        "det_arr": det_arr_ser.tolist()
                })

                ret_not_pass = {
                        "file": filename,
                        "code": 301,
                        "message": "敏感人物",
                        "result": "不合规",
                        "data": data,
                }

        if not bool(ret_not_pass):
            ret = ret_pass
        else:
            ret = ret_not_pass

    return ret


def listsbigger(list_1, list_2):
    result = []
    if len(list_1) == len(list_2):
        for i in range(len(list_1)):
            if list_1[i] < list_2[i]:
                result.append(False)
            else:
                result.append(True)
    return result


def cal_sim_new(emb, emb_data):
    likely = {}
    for i in list(emb_data.keys()):
        emb_feature = emb_data[i]['emb_ave']
        likely[i] = feat_distance_cosine(emb, emb_feature)
    return likely


def feat_distance_cosine(feat1, feat2):
    similarity = np.dot(feat1 / np.linalg.norm(feat1, 2), feat2 / np.linalg.norm(feat2, 2))
    return similarity


def feat_distance_l2(feat1, feat2):
    feat1_norm = feat1 / np.linalg.norm(feat1, 2)
    feat2_norm = feat2 / np.linalg.norm(feat2, 2)
    similarity = 1.0 - np.linalg.norm(feat1_norm - feat2_norm, 2) / 2.0
    return similarity



