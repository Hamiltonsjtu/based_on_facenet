import numpy as np
from scipy import misc
import cv2
import grpc
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def get_face_box(img, channel):

    threshold_value = 0.4
    boxes_list = []
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_ssd'
    request.model_spec.signature_name = 'calculate_BBox'
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img, shape=[1, img.shape[0], img.shape[1], img.shape[2]]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    boxes = np.array(result.outputs['boxes'].float_val).reshape(
        result.outputs['boxes'].tensor_shape.dim[0].size,
        result.outputs['boxes'].tensor_shape.dim[1].size,
        result.outputs['boxes'].tensor_shape.dim[2].size
    )
    scores = np.array(result.outputs['scores'].float_val)

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    face_indice_tmp = np.where(scores > threshold_value)
    face_indice = face_indice_tmp[0]

    im_height, im_width = np.shape(img)[0:2]
    for i in face_indice:
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
        boxes_list.append(np.squeeze(bb))


    return boxes_list


def crop_ssd_face_img(img, bboxes):
    ###
    # margin can be given as 0.15
    # img_size: the size of img
    # image_size: the size of cropped face, recommend 160
    ###

    margin = 0.15
    img_size = np.asarray(img.shape)[0:2]
    image_size = 160
    face_imgs_list = []
    bb_new_list = []
    for _, det in enumerate(bboxes):
        bb = np.zeros(4, dtype=np.int32)
        if margin > 1:
            # =================================================
            # cropped with fixed margin which is used for lfw
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[2] - margin / 2, 0)
            bb[2] = np.minimum(det[1] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        else:
            # =================================================
            # cropped with percentage margin can be used for images download from internet
            width = det[1] - det[0]
            height = det[3] - det[2]
            bb[0] = np.maximum(det[0] - margin * width, 0)
            bb[1] = np.maximum(det[2] - margin * height, 0)
            bb[2] = np.minimum(det[1] + margin * width, img_size[1])
            bb[3] = np.minimum(det[3] + margin * height, img_size[0])
        # print('ssd return window height: {}, width: {}'.format(bb[3] - bb[1], bb[2] - bb[0]))

        #### crop by square box
        center = [float(bb[0] + bb[2]) / 2, float(bb[1] + bb[3]) / 2]
        height = float(bb[3] - bb[1])
        weight = float(bb[2] - bb[0])
        max_size = max([height, weight]) / 2
        img_height = img.shape[0]
        img_width = img.shape[1]
        size_up = img_height - center[1]
        size_down = center[1]
        size_left = center[0]
        size_right = img_width - center[0]

        adjust_size = min([max_size, size_up, size_down, size_left, size_right])
        bb_new = np.array(
            [center[0] - adjust_size, center[1] - adjust_size, center[0] + adjust_size, center[1] + adjust_size],
            dtype=np.int32)
        cropped = img[bb_new[1]:bb_new[3], bb_new[0]:bb_new[2], :]
        # print('adjusted window height: {}, width: {}'.format(bb_new[3] - bb_new[1], bb_new[2] - bb_new[0]))

        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        face_imgs_list.append(scaled)
        bb_new_list.append(list(bb_new))

    return face_imgs_list, bb_new_list


def detect_Face_and_crop(img, FACE_CHANNEL, filename):

    result = {}
    boxes_list = get_face_box(img, FACE_CHANNEL)
    _, bb_new_list = crop_ssd_face_img(img, boxes_list)
    bb_new_list_ser = np.reshape(bb_new_list, (1, np.size(bb_new_list)))
    result['bbox'] = np.squeeze(bb_new_list_ser).tolist()
    return result
