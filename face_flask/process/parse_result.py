#coding:utf-8
from logger import code_message
from logger import flask_logger
logger = flask_logger.get_logger(__name__)

def parse_face_detect_result_dict(result_dict, filename):
    ret_dict = {}
    if result_dict['code'] == code_message.no_face_detect_code:
        ret_dict['code'] = code_message.success_code
        ret_dict['filename'] = filename
        ret_dict['message'] = "不存在政治敏感人物"
        ret_dict['extend_message'] = 'no_face_detected' 
        ret_dict['result'] = "合规"
    else:
        ret_dict['code'] = result_dict['code']
        ret_dict['filename'] = filename
        ret_dict['message'] = "internal error"
        ret_dict['result'] = "异常错误"
    return ret_dict
        
def parse_facenet_result_dict(result_dict, filename):
    
    ret_dict = {}
    if result_dict['code'] == code_message.no_face_detect_code:
        ret_dict['code'] = code_message.success_code
        ret_dict['filename'] = filename
        ret_dict['message'] = "Has no faces in image, IMAGES PASS!"
        ret_dict['result'] = "合规"
    else:
        ret_dict['code'] = result_dict['code']
        ret_dict['filename'] = filename
        ret_dict['message'] = "internal error"
        ret_dict['result'] = "异常错误"
    return ret_dict