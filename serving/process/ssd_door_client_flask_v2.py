from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import os
import cv2 as cv
import time
import shutil


def door_img_pts(img, channel):
    channel = grpc.insecure_channel(channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'door_detection'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY # 加了个tf.saved_model
    request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(img,shape=[1,img.shape[0],img.shape[1],img.shape[2]]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    boxes = np.array(result.outputs['detection_boxes'].float_val).reshape(
                      result.outputs['detection_boxes'].tensor_shape.dim[0].size,
                      result.outputs['detection_boxes'].tensor_shape.dim[1].size,
                      result.outputs['detection_boxes'].tensor_shape.dim[2].size
                  )

    scores = np.array(result.outputs['detection_scores'].float_val)
    detection_classes = np.array(result.outputs['detection_classes'].float_val)


    # num_detections = np.array(result.outputs['num_detections'].float_val)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    height, width = img.shape[:2]

    pts_box =[]  
    pts = None
    door_img = None
    scores_max=0
    detection_class = None
    for i in range(boxes.shape[0]):
        if ( scores[i] > 0.5)  and (scores[i] > scores_max):
            scores_max = scores[i]
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)

            pts = np.array([xmin, ymin, xmax, ymax])

            detection_class = detection_classes[i]

    #channel.close()
    return detection_class
    # return pts
#
# path = "/mnt/AutoROI/"
#
# pic_list = os.listdir(path)
#
# with open("door_information_test.txt",'w') as f:
#     for pic in pic_list:
#         print(pic)
#         img = cv.imread(path + pic)
#         # print(img.shape)
#         # time.sleep(1000)
#         pts = door_img_pts(img,"192.168.1.254:9910")

# print(pts)
# if pts is not None:
#     # print(pts)
#
#     cv.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),(0,0,255),thickness=5)
#     # cv.imwrite("/home/myue/about_door/test/veri_2/" + pic,img)
#     cv.imshow("detection", img)
#     cv.waitKey(0)
# #     f.write(pic + " " +str(pts[0]) +" "+str(pts[1]) +" "+str(pts[2]) +" "+str(pts[3]) + "\n")
# else:
#     f.write(pic + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + "\n")
#
# # # print(pts)

#
#
# img = cv.imread("/home/myue/about_door/test/2.jpg")
# # print(img.shape)
# # time.sleep(1000)
# pts = door_img_pts(img,"192.168.1.254:9910")
# # print(pts)
# if pts is not None:
#     # print(pts)
#
#     cv.rectangle(img,(pts[0],pts[1]),(pts[2],pts[3]),(0,0,255),thickness=5)
#     # cv.imwrite("/home/myue/about_door/test/veri_2/" + pic,img)
#     cv.imshow("detection", img)
#     cv.waitKey(0)


# path = "/mnt/sda2/mengyue/project_data/door_detection/train_dir/door_0524/"
path = "/mnt/AutoROI/door_0524/"

pic_list = os.listdir(path)

for pic in pic_list:
    # print(pic)
    img = cv.imread(path + pic)
    # print(img.shape)
    # time.sleep(1000)
    detection_class = door_img_pts(img, "192.168.1.254:9090")
    # print(detection_class,type(detection_class))

    if detection_class == 1.0:
        # print("close")
        # time.sleep(10000)
        shutil.copy(path + pic , "/mnt/AutoROI/door_0524_close")

    elif detection_class == 2.0:
        # print(pts)
        shutil.copy(path + pic , "/mnt/AutoROI/door_0524_open")

    else:
        shutil.copy(path + pic, "/mnt/AutoROI/door_0524_not_checkout")
