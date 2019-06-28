#coding:utf-8
import cv2
import grpc
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS

from logger import code_message
from logger import flask_logger
logger = flask_logger.get_logger(__name__)


def rescale_points(points, org_H, org_W, min_ratio, delta_h, delta_w ):
    point_x, point_y = points[0], points[1]
    org_point_x = (point_x-delta_w)/min_ratio 
    org_point_y = (point_y-delta_h)/min_ratio 
    return org_point_x, org_point_y


def expand_img(im,  resize_W, resize_H):
    try:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        H, W = im.shape[0], im.shape[1]
        if H > resize_H or W > resize_W:
            scale_ratio_w =  resize_W/W
            scale_ratio_h = resize_H/H
            min_ratio = min(scale_ratio_w, scale_ratio_h)
        else:
            min_ratio = 1
        scale_h = int(H * min_ratio)
        scale_w = int(W * min_ratio)
        new_image = np.zeros((resize_H, resize_W, 3), dtype=np.uint8)
        delta_h = (resize_H - scale_h)//2
        delta_w = (resize_W - scale_w)//2
        im = cv2.resize(im, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        new_image[delta_h:scale_h+delta_h, delta_w:scale_w+delta_w, :] = im
        #return new_image, None 
        return new_image, min_ratio, delta_h, delta_w, None
    except Exception as e:
        logger.error(e)
        return None, None, None, None, str(e)


def get_face_box(img, channel):
    threshold_value = 0.5
    dict_return = {}
    boxes_list = []
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    resize_W = 300
    resize_H = 300
   
    #new_img = cv2.resize(img, (288, 512))
    new_img, min_ratio, delta_h, delta_w, err_msg = expand_img(img,  resize_W, resize_H)
    if new_img is None:
        dict_return['code'] = code_message.source_images_invalid_error_code
        dict_return['message'] = err_msg
        dict_return['data'] = boxes_list
        logger.error("code: %s"%(code_message.source_images_invalid_error_code))
        logger.error("message: %s"%(err_msg))
        logger.error(err_msg)
        return dict_return
    im_width = img.shape[1]
    im_height = img.shape[0]
    #new_img = np.array(new_img).astype(np.float32) / 255.0
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_ssd'
    request.model_spec.signature_name = 'calculate_BBox'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(new_img,shape=[1,new_img.shape[0],new_img.shape[1],new_img.shape[2]]))
    try:
        result = stub.Predict(request, 10.0)  # 10 secs timeout
    except Exception as e:
        dict_return['code'] = code_message.face_detect_grpc_connect_error_code
        dict_return['message'] = str(e)
        dict_return['data'] = boxes_list
        logger.error("code: %s"%(code_message.face_detect_grpc_connect_error_code))
        logger.error("message: %s"%(str(e)))
        logger.error(e)
        return dict_return
    try:
        boxes = np.array(result.outputs['boxes'].float_val).reshape(
                          result.outputs['boxes'].tensor_shape.dim[0].size,
                          result.outputs['boxes'].tensor_shape.dim[1].size,
                          result.outputs['boxes'].tensor_shape.dim[2].size
                      )
    
        scores = np.array(result.outputs['scores'].float_val)
        detection_classes = np.array(result.outputs['classes'].float_val)

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        face_indice_tmp = np.where(scores > threshold_value)
        face_indice = face_indice_tmp[0]
        boxes_tmp = boxes[face_indice, :]
        for i in face_indice:
            box = boxes[i, :]
            _ymin = box[0]
            _xmin = box[1]
            _ymax = box[2]
            _xmax = box[3]
            xmin, ymin = rescale_points([_xmin*resize_W, _ymin*resize_H], im_height, im_width, min_ratio, delta_h, delta_w )
            xmax, ymax = rescale_points([_xmax*resize_W, _ymax*resize_H], im_height, im_width, min_ratio, delta_h, delta_w )
            bb = []
            bb.append(max(xmin-22, 0))   # left
            bb.append(max(ymin-22, 0))  # bottom
            bb.append(min(xmax+22, im_width))   # right
            bb.append(min(ymax+22, im_height))  # top
#             bb.append(max(xmin * im_width-22, 0))   # left
#             bb.append(max(ymin * im_height-22, 0))  # bottom
#             bb.append(min(xmax * im_width+22, im_width))   # right
#             bb.append(min(ymax * im_height+22, im_height))  # top
            boxes_list.append(bb)

        if len(boxes_list) > 0:
            dict_return['code'] = code_message.face_detect_box_code
            dict_return['message'] = code_message.face_detect_box_message
            dict_return['data'] = boxes_list
        else:
            dict_return['code'] = code_message.no_face_detect_code
            dict_return['message'] = code_message.no_face_detect_message
            dict_return['data'] = boxes_list
        return dict_return  
    except Exception as e:
        dict_return['code'] = code_message.face_detect_predict_points_error_code
        dict_return['message'] = str(e)
        dict_return['data'] = boxes_list
        logger.error("code: %s"%(code_message.face_detect_predict_points_error_code))
        logger.error("message: %s"%(str(e)))
        logger.error(e)
        return dict_return


def get_face_box_update(img, channel):
    threshold_value = 0.5
    dict_return = {}
    boxes_list = []
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # new_img = np.array(new_img).astype(np.float32) / 255.0
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_ssd'
    request.model_spec.signature_name = 'calculate_BBox'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, shape=[1, img.shape[0], img.shape[1], img.shape[2]]))
    try:
        result = stub.Predict(request, 10.0)  # 10 secs timeout
    except Exception as e:
        dict_return['code'] = code_message.face_detect_grpc_connect_error_code
        dict_return['message'] = str(e)
        dict_return['data'] = boxes_list
        logger.error("code: %s" % (code_message.face_detect_grpc_connect_error_code))
        logger.error("message: %s" % (str(e)))
        logger.error(e)
        return dict_return

    try:
        boxes = np.array(result.outputs['boxes'].float_val).reshape(
            result.outputs['boxes'].tensor_shape.dim[0].size,
            result.outputs['boxes'].tensor_shape.dim[1].size,
            result.outputs['boxes'].tensor_shape.dim[2].size
        )
        scores = np.array(result.outputs['scores'].float_val)
        # detection_classes = np.array(result.outputs['classes'].float_val)

        boxes = np.squeeze(boxes)


        scores = np.squeeze(scores)
        face_indice_tmp = np.where(scores > threshold_value)
        face_indice = face_indice_tmp[0]
        boxes_ = boxes[face_indice, :]

        im_height, im_width = np.shape(img)[0:2]
        for i in face_indice:
            # print('the {}-th face'.format(i))
            # print('{} box: {} and score {}'.format(i, boxes[i,:], scores[i]))
            box = boxes[i, :]
            ymin = box[0]
            xmin = box[1]
            ymax = box[2]
            xmax = box[3]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = xmin * im_width   # left
            bb[1] = xmax * im_width   # right
            bb[2] = ymin * im_height  # bottom
            bb[3] = ymax * im_height  # top
            boxes_list.append(bb)

        # print('-------------- ssd output --------------')
        # print(boxes_list)

        if len(boxes_list) > 0:
            dict_return['code'] = code_message.face_detect_box_code
            dict_return['message'] = code_message.face_detect_box_message
            dict_return['data'] = boxes_list
        else:
            dict_return['code'] = code_message.no_face_detect_code
            dict_return['message'] = code_message.no_face_detect_message
            dict_return['data'] = boxes_list
        return dict_return
    except Exception as e:
        dict_return['code'] = code_message.face_detect_predict_points_error_code
        dict_return['message'] = str(e)
        dict_return['data'] = boxes_list
        logger.error("code: %s" % (code_message.face_detect_predict_points_error_code))
        logger.error("message: %s" % (str(e)))
        logger.error(e)
        return dict_return