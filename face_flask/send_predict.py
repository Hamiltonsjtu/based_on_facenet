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


def get_faces_and_predict(img, face_channel, facenet_channel, DATA, filename):
    face_dict = ssd_face_detect_client.get_face_box_update(img, face_channel)
    if face_dict['code'] == code_message.face_detect_box_code:
        boxes_img, boxes_size = face_util.crop_ssd_face_img_update(img, face_dict['data'], filename)
        det_arr_ser = np.reshape(boxes_size, (1, np.size(boxes_size)))
        ret_dict = {}
        ret = {}
        maximum = []
        maximum_name = []
        for faces in boxes_img:
            faceNet_dict = faceNet_client.img_to_emb_feature(faces, facenet_channel)
            if faceNet_dict['code'] != code_message.facenet_predict_success_code:
                continue
            emb = faceNet_dict['data']
            num_face = int(len(emb)/128)
            for i in range(num_face):
                emb_face = emb[i*128:(1+i)*128]
                likely = cal_sim_updata(emb_face, DATA)
                ret[str(i)] = likely
                maximum_name.append(max(likely, key=likely.get))
                maximum.append(likely[max(likely, key=likely.get)])
        # threshold for recognition
        threshold_value = 0.80
        if len(boxes_img) != len(maximum):
            ret_dict['filename'] = filename
            ret_dict['code'] = code_message.facenet_image_grpc_error_code
            ret_dict['message'] = "internal_error"
            ret_dict['result'] = "异常错误"

        elif max(maximum) < threshold_value:
            index = list(np.where(np.array(maximum) > threshold_value)[0])
            det_arr_tmp = np.array(boxes_size)[index, :]
            det_arr_ser = np.reshape(det_arr_tmp, (1, np.size(det_arr_tmp)))

            ret_dict['filename'] = filename
            ret_dict['code'] = code_message.success_code
            ret_dict['message'] = "Has_face_pass"
            ret_dict['result'] = "合规"
            ret_dict["det_arr"] = det_arr_ser.tolist()
        else:
            index = list(np.where(np.array(maximum) > threshold_value)[0])
            det_arr_tmp = np.array(boxes_size)[index, :]
            det_arr_ser = np.reshape(det_arr_tmp, (1, np.size(det_arr_tmp)))

            print('num of faces who is sensitive: ', index)
            data = []
            for i in index:
                data.append(
                    {
                        "face_id": str(i),
                        "user_name": maximum_name[i],
                        "score": maximum[i]
                    })

            ret_dict['filename'] = filename
            ret_dict['code'] = code_message.success_code
            ret_dict['message'] = "敏感人物"
            ret_dict['result'] = "不合规"
            ret_dict['data'] = data
            ret_dict["det_arr"] = det_arr_ser.tolist()
        return ret_dict
    else:
        face_result_dict = parse_result.parse_face_detect_result_dict(face_dict, filename)
        return face_result_dict


def cal_sim(emb, DATA):
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

    emb_jack = emb
    jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
    likely = {}
    txt = {}
    for i in people:
        jack_dist[i]['dist_average'] = [
            np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
        jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
        jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean']) / (
        emb_data[i]['log_dist_std'])
#         jack_dist[i]['Prob'] = st.norm.pdf(jack_dist[i]['Z_value']) / st.norm.pdf(0)
#         likely[i] = jack_dist[i]['Prob'][0]
    
        jack_dist[i]['Prob'] = feat_distance_cosine(emb_jack, emb_data[i]['average_emb'])  
        # jack_dist[i]['Prob'] = feat_distance_l2(emb_jack, emb_data[i]['average_emb'])
        likely[i] = jack_dist[i]['Prob']
    return likely


def cal_sim_updata(emb, DATA):
    likely = {}
    peoples = [i for i, _ in DATA.items()]
    for i in peoples:
        likely[i] = feat_distance_cosine(emb, DATA[i])
    return likely


def feat_distance_cosine(feat1, feat2):
    similarity = np.dot(feat1 / np.linalg.norm(feat1, 2), feat2 / np.linalg.norm(feat2, 2))
    return similarity
