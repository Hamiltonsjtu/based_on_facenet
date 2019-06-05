
from __future__ import print_function
import cv2 as cv
import numpy as np
import sys
import threading
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import os
import time


def ssd_pts(img, channel):
    # channel = grpc.insecure_channel(channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'object_detection'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY # 加了个tf.saved_model
    request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(img,shape=[1,img.shape[0],img.shape[1],img.shape[2]]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    # print("========", result)
    boxes = np.array(result.outputs['detection_boxes'].float_val).reshape(
                      result.outputs['detection_boxes'].tensor_shape.dim[0].size,
                      result.outputs['detection_boxes'].tensor_shape.dim[1].size,
                      result.outputs['detection_boxes'].tensor_shape.dim[2].size
                  )

    scores = np.array(result.outputs['detection_scores'].float_val)
    # num_detections = np.array(result.outputs['num_detections'].float_val)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    height, width = img.shape[:2]

    # print(boxes.shape)  ##(100, 4)
    pts = None
    scores_max=0
    b, g, r = cv.split(img)
    # print("gray")
    pts_box = {}
    pts_box_exce = {}
    pts_box_exce['brightness'] = 0
    pts_box_exce['pts']= None
    pts_box_exce['score']= 0
    # print(boxes.shape[0])
    # print(scores[0])
    W = 0
    H = 0
    for i in range(boxes.shape[0]):
        if ( scores[i] > 0.05 ):
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)

            W = xmax-xmin
            H = ymax-ymin

            img_warn = img[ymin:ymax,xmin:xmax]
            brightness = np.sum(img_warn)
            x1 = xmin
            y1 = ymin
            x2 = (xmax + xmin) / 2
            y2 = ymin - (ymax - ymin)
            x3 = xmax
            y3 = ymin
            x4 = xmax
            y4 = ymax
            x5 = (xmax + xmin) / 2
            y5 = ymax + (ymax - ymin)
            x6 =  xmin
            y6 = ymax


            ## 往外扩充边长的1/11
            x11 = xmin - (xmax - xmin)/11
            x61 = xmin - (xmax - xmin)/11
            x31 = xmax + (xmax - xmin)/11
            x41 = xmax + (xmax - xmin)/11
            # print("length: ", xmax-xmin)


            # pts = np.array([[x1 , y1 ], [x2 , y2 ], [x3 , y3 ],
            #                 [x4 , y4 ],
            #                 [x5 , y5 ], [x6 , y6 ]], np.int32)

            pts = np.array([[x11, y1], [x2, y2], [x31, y3],
                            [x41, y4],
                            [x5, y5], [x61, y6]], np.int32)

            pts = pts.reshape((-1, 1, 2))

            pts_box["brightness"] = brightness
            pts_box["pts"] = pts
            pts_box['score'] = scores[i]
            if (b==g).all() and (b==r).all():
                #print("gray")
                if pts_box['brightness'] > pts_box_exce['brightness']:
                    #print("score: ", scores[i])
                    pts_box_exce['brightness'] = pts_box["brightness"]
                    pts_box_exce['pts'] = pts_box["pts"]
            else:
                #print("color")
                if pts_box_exce['pts'] is None:
                    pts_box_exce['pts'] = pts_box["pts"]
                elif pts_box['pts'][0][0][1] < pts_box_exce['pts'][0][0][1]:
                    # pts_box_exce['score'] = pts_box["score"]
                    pts_box_exce['pts'] = pts_box["pts"]

    # print(pts_box_exce['pts'])
    if W < H:
        return (pts_box_exce['pts'],"single")
    else:
        return (pts_box_exce['pts'],"None")