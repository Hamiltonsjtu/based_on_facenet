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
def _Onet(weight_path = 'model48.h5'):
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
def _Rnet(weight_path = 'model24.h5'):
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
class _PNET_post(Layer):
    def __init__(self, **kwargs):
        super(_PNET_post, self).__init__(**kwargs)
        self.scales     = 1.0
        self.threshold  = 0.6
    def tffix(self, mat):
        mask1 = mat >= 0
        a_floor = tf.math.floor(mat)
        b_ceil = tf.math.ceil(mat)
        remat = tf.where(mask1, a_floor, b_ceil)
        return tf.cast(remat, dtype=tf.float32)
    def rec2square(self, rectangles):
        w = tf.subtract(tf.gather(rectangles, 2, axis=-1), tf.gather(rectangles, 0, axis=-1))
        h = tf.subtract(tf.gather(rectangles, 3, axis=-1), tf.gather(rectangles, 1, axis=-1))
        l = tf.maximum(w, h)
        rec_1 = tf.gather(rectangles,0, axis=-1) + w*0.5 - l*0.5
        rec_2 = tf.gather(rectangles,1, axis=-1) + h*0.5 - l*0.5
        rec_3 = rec_1 + l
        rec_4 = rec_2 + l
        rec_5 = tf.gather(rectangles,4, axis=-1)
        rectangles = tf.transpose(tf.concat([[rec_1], [rec_2], [rec_3], [rec_4], [rec_5]], axis=0), perm=[1,0])
        return rectangles
    def cls_bb2rect(self, Cls_pro, Roi):
        scale = self.scales
        width = tf.shape(Cls_pro)[0]
        height = tf.shape(Cls_pro)[1]
        out_side = tf.maximum(width, height)
        in_side = 2 * out_side + 11
        cond = tf.not_equal(2 * out_side, 11)
        ftrue = tf.cast((in_side - 12) / (out_side - 1), dtype=tf.float32)
        ffals = tf.constant(0.0, dtype=tf.float32)
        stride = tf.cond(cond, lambda: ftrue, lambda: ffals)
        threshold_p = self.threshold
        Cls_pro_coor = tf.gather(Cls_pro, 0, axis=2)
        thresh_index = tf.where(Cls_pro_coor >= threshold_p)
        boundingbox = tf.cast(tf.slice(thresh_index, [0, 0], [-1, 2]), dtype=tf.float32)
        bb1_tmp = tf.math.scalar_mul(stride, boundingbox)
        bb2_tmp = tf.math.scalar_mul(stride, boundingbox) + 11.0
        bb1 = self.tffix(bb1_tmp)
        bb2 = self.tffix(bb2_tmp)
        boundingbox = tf.concat([bb1, bb2], -1)
        dx1 = tf.squeeze(tf.gather_nd(tf.gather(Roi, 0), [thresh_index]))
        dx2 = tf.squeeze(tf.gather_nd(tf.gather(Roi, 1), [thresh_index]))
        dx3 = tf.squeeze(tf.gather_nd(tf.gather(Roi, 2), [thresh_index]))
        dx4 = tf.squeeze(tf.gather_nd(tf.gather(Roi, 3), [thresh_index]))
        compare = tf.gather(Roi, 0)
        Cls_pro_GA = tf.gather(Cls_pro, [0], axis=2)
        score = tf.expand_dims(tf.squeeze(tf.gather_nd(Cls_pro_GA, [thresh_index])), axis=1)
        offset = tf.squeeze(tf.stack([dx1, dx2, dx3, dx4], axis=-1))
        boundingbox = tf.squeeze(boundingbox + offset * 12.0 * scale)
        rectangles = tf.concat([boundingbox, score], axis=-1)
        rectangles = self.rec2square(rectangles)
        return rectangles
    def post_data_Pnet(self, classifier, bbox_regress, img_shape):
        Cls_pro = tf.transpose(tf.gather(classifier, [1], axis=2), perm=[1, 0, 2])
        Roi = tf.transpose(tf.slice(bbox_regress, [0,0,0], [-1,-1,-1]), perm=[2, 1, 0])
        rectangles = self.cls_bb2rect(Cls_pro, Roi)
        img_shp = img_shape + 1.0
        width = img_shape[0]
        height = img_shape[1]
        # pick = []
        # for i in range(tf.shape[1]):
        #     x1 = tf.cast(tf.maximum(0, rectangles[0][i][0]), dtype=tf.int32)
        #     y1 = tf.cast(tf.maximum(0, rectangles[0][i][1]), dtype=tf.int32)
        #     x2 = tf.cast(tf.minimum(width, rectangles[0][i][2]), dtype=tf.int32)
        #     y2 = tf.cast(tf.minimum(height, rectangles[0][i][3]), dtype=tf.int32)
        #     sc = rectangles[0][i][4]
        #
        #     if x2>x1 and y2>y1:
        #         pick = Concatenate(axis=-2)(pred_masks)
        return rectangles, rectangles, img_shp

    def call(self, inputs):
        classifier = inputs[0]
        bbox_regress = inputs[1]
        img_shape = inputs[2]
        batch_data = (classifier, bbox_regress, img_shape)
        bb1, bb2, bb3 = tf.map_fn(lambda x: self.post_data_Pnet(x[0], x[1], x[2]), batch_data, dtype=(tf.float32, tf.float32, tf.float32))
        return [bb1, bb2, bb3]

    # def compute_output_shape(self, input_shape):
    #     shape = tf.TensorShape(input_shape).as_list()
    #     return tf.TensorShape(shape)

def _Pnet(weight_path = 'model12.h5'):
    input_img = Input(shape=[None, None, 3])
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input_img)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    def img_shape(input_img_0):
        return tf.map_fn(lambda xxx: tf.cast(tf.shape(xxx)[0:2], tf.float32), input_img_0)

    shape_img = Lambda(lambda xx: img_shape(xx), name= 'img_shape')(input_img)
    ### ----- ADD LAYER  ----  ###
    outs = _PNET_post(name='Pnet_output_post')([classifier, bbox_regress, shape_img])
    model = Model(inputs = [input_img], outputs = outs)
    ###  LOAD TRAINED WEIGHTS
    model.load_weights(weight_path, by_name=True)
    return model
Pnet = _Pnet(r'12net.h5')
# Rnet = _Rnet(r'24net.h5')
# Onet = _Onet(r'48net.h5')
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
    # origin_h, origin_w, ch = img.shape
    img = np.expand_dims(img, axis=0)
    # shape_img = [origin_w, origin_h]
    # shape_img = np.expand_dims(np.array(shape_img), axis=0)
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
    #         for i in range(5, 15, 2):
    #             cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
    # cv2.imshow("test", draw)
    # c = cv2.waitKey(1) & 0xFF
    # print(c)
    # if c == 27 or c == ord('q'):
    #     break
    # cv2.imwrite('test.jpg', draw)
    # pnet_layer_output = get_layer_output(img, Pnet, 'Pnet_output_post')
    output = Pnet.predict(img)
    # cls_pro, roi = Pnet_out(ouput)
    # model = Model(ouput, pro_bbox)
    # image_scaled = model.predict(img)
    print('Done')


