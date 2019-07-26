# ! /usr/bin/env python
# coding=utf-8
import os
import time
import numpy as np
import tensorflow as tf
import grpc
import requests
import time
import cv2
import scipy.misc as misc
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from beautifultable import BeautifulTable

# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('facenet', '192.168.1.252:8500', 'PredictionService host:port')
# FACENET_CHANNEL = grpc.insecure_channel(FLAGS.facenet)

image_dir = r'F:\TEST_crop'
image_path = [image_dir + '/' + i for i in os.listdir(image_dir)]
img_LIST = []


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


for path in image_path:
    img = misc.imread(os.path.expanduser(path), mode='RGB')
    aligned = misc.imresize(img, (160, 160), interp='bilinear')
    # image = np.expand_dims(image, axis=0)
    prewhitened = prewhiten(aligned)
    img_LIST.append(prewhitened)
    images = np.stack(img_LIST)
print('image size', np.shape(images))

channel = grpc.insecure_channel('192.168.1.254:9001')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# Send request
# See prediction_service.proto for gRPC request/response details.
request = predict_pb2.PredictRequest()
request.model_spec.name = 'facenet'
request.model_spec.signature_name = 'calculate_embeddings'
# print('request ', request)
request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(images, shape=np.shape(images), dtype=tf.float32))
request.inputs['phase'].CopyFrom(tf.contrib.util.make_tensor_proto(False))

print('========================')
result_tmp = stub.Predict(request, 10.0)  # 10 secs timeout
# results = stub.Predict(request, 10.0)
# embds = result_tmp.outputs._values['embeddings']
embds = list(result_tmp.outputs['embeddings'].float_val)
faces_num = len(embds) // 128
embs = np.zeros((faces_num, 128))
for i in range(faces_num):
    embs[i, :] = embds[i*128:(i+1)*128]


dist = np.zeros((faces_num, faces_num))
def distance(embeddings1, embeddings2):
    dot = np.dot(embeddings1, embeddings2)
    norm = np.linalg.norm(embeddings1, ord=2) * np.linalg.norm(embeddings2, ord=2)
    similarity = dot / norm
    if similarity > 1:
        similarity = 1.0
    return similarity

for i in range(faces_num):
    for j in range(faces_num):
        embedding_1 = embs[i, :]
        embedding_2 = embs[j, :]
        dist[i,j] = distance(embedding_1, embedding_2)

table = BeautifulTable()
# table.column_headers = data_keys
for i in range(faces_num):
    table.append_row(dist[i,:])
print(table)





# for i in range(len(data_keys)):
#     for j in range(len(data_keys)):
#         embedding_1 = data[data_keys[i]]
#         embedding_2 = data[data_keys[j]]
#         dist[i,j] = distance(embedding_1, embedding_2)
#
# table = BeautifulTable()
# table.column_headers = data_keys
# for i in range(len(data_keys)):
#     table.append_row(dist[i,:])
# print(table)