#coding:utf-8
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
import cv2
from logger import code_message
from logger import flask_logger
logger = flask_logger.get_logger(__name__)


def resize_image(im, resize_W, resize_H):
    try:
        H, W = im.shape[0], im.shape[1]
        scale_ratio_w = resize_W/W
        scale_ratio_h = resize_H/H
        min_ratio = min(scale_ratio_w, scale_ratio_h)
        scale_h = int(H * min_ratio)
        scale_w = int(W * min_ratio)
        new_image = np.zeros((resize_H, resize_W, 3), dtype=np.float32)
        delta_h = (resize_H - scale_h)//2
        delta_w = (resize_W - scale_w)//2
        im = cv2.resize(im, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        new_image[delta_h:scale_h+delta_h, delta_w:scale_w+delta_w, :] = im
        return new_image, None
    except Exception as e:
        logger.error(e)
        return None, str(e)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_face(image):
    prewhitened = prewhiten(image)
    images = np.stack(prewhitened)
    return images


def img_to_emb_feature(img, channel):
    #img = cv2.resize(img, (160, 160))
    feature_list = []
    dict_return = {}
    img = load_face(img)
    img, err_msg = resize_image(img, 160, 160)
    if img is None:
        dict_return['code'] = code_message.facenet_images_invalid_error_code
        dict_return['message'] = err_msg
        dict_return['data'] = feature_list
        logger.error("code: %s"%(code_message.facenet_images_invalid_error_code))
        logger.error("message: %s"%(err_msg))
        logger.error(err_msg)
        return dict_return
    try:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'facenet'
        request.model_spec.signature_name = 'calculate_embeddings'
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(img, shape=[1, img.shape[0], img.shape[1],img.shape[2]], dtype=tf.float32))
        request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
        result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout
        result = result_tmp.outputs['embeddings'].float_val
        feature_list = list(result)
        dict_return['code'] = code_message.facenet_predict_success_code
        dict_return['message'] = err_msg
        dict_return['data'] = feature_list
        logger.error("code: %s"%(code_message.facenet_predict_success_message))
        logger.error("message: %s"%(err_msg))
        logger.error(err_msg)
        return dict_return
    except Exception as e:
        dict_return['code'] = code_message.facenet_grpc_connect_error_code
        dict_return['message'] = str(e)
        dict_return['data'] = feature_list
        logger.error("code: %s"%(code_message.facenet_grpc_connect_error_code))
        logger.error("message: %s"%(str(e)))
        logger.error(err_msg)
        return dict_return
