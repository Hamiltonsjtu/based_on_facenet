
from __future__ import print_function

# This is a placeholder for a Google-internal import.

import cv2
import grpc
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
from scipy import misc
import os
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', '192.168.1.254:9003',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

IMAG_DIR = r'E:\shuai\GOOGLE\google_images_download\downloads'

def cropsize(det, margin, img_size):
    det = np.squeeze(det)
    bb = np.zeros(4, dtype=np.int32)
    if margin > 1:
        # =================================================
        # cropped with fixed margin which is used for lfw
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    else:
        # =================================================
        # cropped with percentage margin can be used for images download from internet
        width = det[2] - det[0]
        height = det[3] - det[1]
        bb[0] = np.maximum(det[0] - margin * width, 0)
        bb[1] = np.maximum(det[1] - margin * height, 0)
        bb[2] = np.minimum(det[2] + margin * width, img_size[1])
        bb[3] = np.minimum(det[3] + margin * height, img_size[0])

    return bb


def main():
    peoples_dir = os.listdir(IMAG_DIR)
    for people_dir in peoples_dir:
        kk = 0
        people_abs_dir = os.path.join(IMAG_DIR, people_dir)
        for i_img in os.listdir(people_abs_dir):
            img_name, img_extension = os.path.splitext(i_img)
            if i_img.endswith('.jpg') or i_img.endswith('.png'):
                img_path = os.path.join(people_abs_dir, i_img)
                try:
                    image_tmp = misc.imread(img_path, mode='RGB')
                except:
                    continue
                img = cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB)
                print('people_dir: {},  image size: {}'.format(people_dir, img.shape))
                threshold_value = 0.7
                channel = grpc.insecure_channel(FLAGS.server)
                stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'face_ssd'
                request.model_spec.signature_name = 'calculate_BBox'
                request.inputs['images'].CopyFrom(
                tf.contrib.util.make_tensor_proto(img, shape=[1, img.shape[0], img.shape[1], img.shape[2]]))
                try:
                    result = stub.Predict(request, 10.0)  # 10 secs timeout
                except:
                    continue
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
                    # print('face #{} bounding box {}'.format(i, bb))
                    # img = cv2.rectangle(img, (bb[0], bb[2]), (bb[1], bb[3]), (0, 255, 0))
                    dst = people_abs_dir+'_crop'
                    if not os.path.exists(dst):
                        os.mkdir(dst)

                    bb = cropsize([bb[0], bb[2], bb[1], bb[3]], 0.15, img.shape[0:2])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                    cropped_resize = cv2.resize(cropped, (160, 160)) # resize
                    # cropped_resize = resize_crop(cropped, 160)     # resize by fill black background

                    cv2.imwrite(dst+'/'+str(kk)+'_'+str(i)+img_extension, cropped_resize)
                    kk += 1

def resize_crop(cropped, dimsize):
    height, width, chn = cropped.shape[0], cropped.shape[1], cropped.shape[2]
    max_h_w = max(height, width)
    ratio = dimsize / max_h_w
    h_n = int(height * ratio)
    w_n = int(width * ratio)
    img = cv2.resize(cropped, (w_n, h_n))
    top = int(np.floor((dimsize - h_n) / 2))
    bottom = int(np.ceil((dimsize - h_n) / 2))
    left = int(np.floor((dimsize - w_n) / 2))
    right = int(np.ceil((dimsize - w_n) / 2))
    img_n = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(img_n, (dimsize, dimsize))

if __name__ == '__main__':
    # tf.app.run()
    main()
