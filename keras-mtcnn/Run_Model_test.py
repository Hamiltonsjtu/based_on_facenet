import sys
import tools_matrix as tools
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import keras.backend as K
from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute, Lambda
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
from keras.layers import Layer, Concatenate, Reshape


def _Onet( weight_path = 'model48.h5'):
    input = Input(shape = [48,48,3])
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)
    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


def _Rnet (weight_path = 'model24.h5'):
    input = Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

class _PNET(Layer):
    def __init__(self, **kwargs):
        super(_PNET, self).__init__(**kwargs)
        self.scales     = 1.0
        self.threshold  = 0.6

    def tffix(self, tensor):
        mask1 = tensor >= 0
        tensor = tf.where(mask1, lambda: tf.floor(tensor), lambda : tf.ceil(tensor))
        return tensor
    def call(self, inputs):
        classifier = inputs[0]
        bbox_regress = inputs[1]
        Cls_pro = tf.transpose(tf.slice(classifier, [0,0,0,1], [-1,-1,-1,1]), perm=[0, 2, 1, 3])
        Roi = tf.transpose(tf.slice(bbox_regress, [0,0,0,0], [-1,-1,-1,-1]), perm=[0, 3, 2, 1])
        scale = self.scales
        width = tf.shape(Cls_pro)[1]
        height = tf.shape(Cls_pro)[2]
        out_side = tf.maximum(width, height)
        in_side = 2*out_side + 11
        cond = tf.not_equal(2*out_side, 11)
        ftrue = tf.cast((in_side - 12)/(out_side -1), dtype=tf.float32)
        ffals = tf.constant(0.0, dtype=tf.float32)
        stride = tf.cond(cond, lambda: ftrue, lambda: ffals)
        threshold_p = self.threshold

        thresh_index = tf.where(Cls_pro >= threshold_p)
        boundingbox = tf.expand_dims(tf.slice(thresh_index, [0, 1], [-1, 2]), axis=0)
        bb1_tmp = tf.math.scalar_mul(stride, tf.cast(boundingbox, dtype=tf.float32))
        bb2_tmp = tf.math.scalar_mul(stride, tf.cast(boundingbox, dtype=tf.float32))
        bb1 = self.tffix(bb1_tmp)
        bb2 = self.tffix(bb2_tmp)
        return boundingbox


def _Pnet(weight_path = 'model12.h5'):
    input = Input(shape=[None, None, 3])
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    ### ----- ADD LAYER  ----  ###
    outs = _PNET(name='Pnet_outs')([classifier, bbox_regress])
    model = Model(inputs=[input], outputs=[outs])

    ###  LOAD TRAINED WEIGHTS
    model.load_weights(weight_path, by_name=True)
    return model


Pnet = _Pnet(r'12net.h5')
Rnet = _Rnet(r'24net.h5')
Onet = _Onet(r'48net.h5')


threshold = [0.6,0.6,0.7]
# video_path = 'WalmartArguments_p1.mkv'
# cap = cv2.VideoCapture(video_path)


def resize_img(x):
    origin_h = tf.cast(tf.shape(x)[1], dtype=tf.float32, name='height')
    origin_w = tf.cast(tf.shape(x)[2], dtype=tf.float32, name='width')
    hs = tf.cast(tf.multiply(origin_h, tf.constant(3.0)), dtype=tf.int32)
    ws = tf.cast(tf.multiply(origin_w, tf.constant(3.0)), dtype=tf.int32)
    output = tf.image.resize_images(x, [ws, hs],
                  align_corners=False,
                  preserve_aspect_ratio=False,
                  name='scaled_img')
    return output


def Pnet_out(x):
    bbox = x[0]
    pro = x[1]
    cls_pro = tf.gather(tf.slice(pro, [0,0,0,1], [1,-1,-1,1]), [0])
    roi = tf.gather(tf.slice(bbox, [0,0,0,0], [1,-1,-1,-1]), [0])

    return cls_pro, roi


def get_layer_output(img, model, layer_name):
    layer = model.get_layer(name=layer_name)
    # functor = K.function([model.input], [layer.get_output_at(0)])
    functor = K.function([model.input], [layer.get_output_at(0)])
    # functor = K.function([model.input], [layer.output])
    outputs = functor([img])[0]
    return outputs


while (True):
    # ret, img = cap.read()
    print('Load picture again!')
    img = cv2.imread(r'F:\TEST\3.png')
    img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = img.shape
    img = np.expand_dims(img, axis=0)
    ###   -------------------   ###
    ###   scale by tensorflow   ###
    ###   -------------------   ###

    # tf_img_in = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
    # scale = 3
    # tf_img_op1 = tf.image.resize_images(tf_img_in, [h * scale, w * scale], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # img_op1 = sess.run([tf_img_op1], feed_dict={tf_img_in: img})

    ###   -------------------   ###
    ###   scale by keras model  ###
    ###   -------------------   ###

    # input_img = Input(shape=[None, None, 3], name='img')
    # Scale_layer = Lambda(resize_img, name='lambda_scale')(input_img)
    # model = Model(input_img, Scale_layer)
    # model.summary()
    # img = np.expand_dims(img, axis=0)
    # image_scaled = model.predict(img)
    # image_height = get_layer_output(img, model, 'lambda_scale')
    # print('Done')
    # for rectangle in rectangles:
    #
    #     if rectangle is not None:
    #         W = -int(rectangle[0]) + int(rectangle[2])
    #         H = -int(rectangle[1]) + int(rectangle[3])
    #         paddingH = 0.01 * W
    #         paddingW = 0.02 * H
    #         crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
    #         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    #         if crop_img is None:
    #             continue
    #         if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
    #             continue
    #         cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
    #
    #         for i in range(5, 15, 2):
    #             cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
    # cv2.imshow("test", draw)
    # c = cv2.waitKey(1) & 0xFF
    # print(c)
    # if c == 27 or c == ord('q'):
    #     break

    # cv2.imwrite('test.jpg', draw)

    output = Pnet.predict(img)
    # cls_pro, roi = Pnet_out(ouput)
    # model = Model(ouput, pro_bbox)
    # image_scaled = model.predict(img)
    print('Done')


