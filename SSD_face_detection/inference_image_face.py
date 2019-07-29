
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
from scipy import misc
import os
import dlib
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import patches, lines
import matplotlib.pyplot as plt


sys.path.append("..")

from SSD_face_detection.utils import label_map_util
from SSD_face_detection.utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

img_path = '../xi_hu.jpg'
image_tmp = misc.imread(os.path.expanduser(img_path), mode='RGB')
image = cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB)
# def random_colors(N, bright=True):
#     """
#     Generate random colors.
#     To get visually distinct colors, generate them in HSV space then
#     convert to RGB.
#     """
#     brightness = 1.0 if bright else 0.7
#     hsv = [(i / N, 1, brightness) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#     random.shuffle(colors)
#     return colors

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=detection_graph, config=config) as sess:
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # print('===========image expanded shape=============')
        # print(np.shape(image_np_expanded))
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        face_indice_tmp = np.where(scores > 0.2)
        face_indice = face_indice_tmp[0]
        # print('type of face_indice {} and shape {}'.format(type(face_indice), np.shape(face_indice)))
        # print(len(face_indice))
        # print(face_indice)

        im_height, im_width = np.shape(image)[0:2]
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
            img_crop = cv2.rectangle(image, (bb[0], bb[2]), (bb[1], bb[3]), (0,255,0))
            cv2.imshow('img_crp', img_crop)
        cv2.waitKey()



