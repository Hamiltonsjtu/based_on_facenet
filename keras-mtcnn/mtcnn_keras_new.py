import sys
import tools_matrix as tools
import cv2
import matplotlib.pyplot as plt
import time
from keras.models import load_model
import numpy as np
import keras as KK
import keras.backend as K
from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute, Lambda, concatenate
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
from keras.layers import Layer, Concatenate, Reshape
def _Pnet( weight_path='./12net.h5'):
    img = Input(shape=[None, None, 3])
    scale = Input(shape=(1,))
    height_raw = Input(shape=(1,))
    width_raw = Input(shape=(1,))
    x_0 = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(img)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU1')(x_0)
    x_0 = MaxPool2D(pool_size=2)(x_0)
    x_0 = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x_0)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU2')(x_0)
    x_0 = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x_0)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU3')(x_0)
    classifier_0 = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x_0)
    bbox_regress_0 = Conv2D(4, (1, 1), name='conv4-2')(x_0)
    outs = Pnet_post(name='Pnet_output_post')([classifier_0, bbox_regress_0, scale, height_raw, width_raw])
    model = Model(inputs=[img, scale, height_raw, width_raw], outputs=[outs])
    model.load_weights(weight_path, by_name=True)
    return model
def _Rnet(weight_path = './24net.h5'):
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
def _Onet(weight_path = './48net.h5'):
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
class _Rnet_post(Layer):
    def __init__(self, **kwargs):
        super(_Rnet_post, self).__init__(**kwargs)
        self.threshold = 0.7
    def _post_data_Rnet(self, classifier, bbox_regress, input_rects, input_height, input_width):
        return 1.0
    def call(self,inputs):
        classifier = inputs[0]
        bbox_regress = inputs[1]
        input_rects = inputs[2]
        input_height = inputs[3]
        input_width = inputs[4]
        batch_data = (classifier, bbox_regress, input_rects)
        bb1, bb2, bb3 = tf.map_fn(lambda x: self.post_data_Rnet(x[0], x[1], x[2], input_height, input_width), batch_data, dtype=(tf.float32, tf.float32, tf.float32))
        return bb1, bb2, bb3
def _Rnet_1(weight_path = 'model24.h5'):
    input = Input(shape=[24, 24, 3]) # change this shape to [None,None,3] to enable arbitraty shape input
    input_rects = Input(shape=[None, 4])
    input_height = Input(shape=(1,))
    input_width = Input(shape=(1,))
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
    R_out = _Rnet_post([classifier, bbox_regress, input_rects, input_height, input_width])
    model = Model([input], [classifier, bbox_regress, input_rects, input_height, input_width])
    model.load_weights(weight_path, by_name=True)
    return model
class Pnet_post(Layer):
    def __init__(self, **kwargs):
        super(Pnet_post, self).__init__(**kwargs)
        # self.scale = 1.0
        self.threshold  = 0.6
    def _tffix(self, mat):
        mask1 = mat >= 0
        a_floor = tf.math.floor(mat)
        b_ceil = tf.math.ceil(mat)
        remat = tf.where(mask1, a_floor, b_ceil)
        return tf.cast(remat, dtype=tf.float32)
    def rec2square(self, rectangles):
        w = tf.subtract(tf.gather(rectangles, 2, axis=-1), tf.gather(rectangles, 0, axis=-1))
        h = tf.subtract(tf.gather(rectangles, 3, axis=-1), tf.gather(rectangles, 1, axis=-1))
        l = tf.maximum(w, h)
        rec_1 = tf.gather(rectangles,0, axis=1) + w*0.5 - l*0.5
        rec_2 = tf.gather(rectangles,1, axis=1) + h*0.5 - l*0.5
        rec_3 = rec_1 + l
        rec_4 = rec_2 + l
        rec_5 = tf.gather(rectangles,4, axis=-1)
        rectangles = tf.transpose(tf.concat([[rec_1], [rec_2], [rec_3], [rec_4], [rec_5]], axis=0), perm=[1,0])
        return rectangles
    def _cls_bb2rect(self, Cls_pro, Roi, scale):
        ## cls: w, h
        ## roi: w, h
        width = tf.shape(Cls_pro)[0]
        height = tf.shape(Cls_pro)[1]
        out_side = tf.maximum(width, height)
        in_side = 2 * out_side + 11
        cond = tf.not_equal(out_side, 1)
        ftrue = tf.cast((in_side - 12) / (out_side - 1), dtype=tf.float32)
        ffals = tf.constant(0.0, dtype=tf.float32)
        stride = tf.cond(cond, lambda: ftrue, lambda: ffals)
        threshold_p = self.threshold
        Cls_pro_coor = tf.gather(Cls_pro, 0, axis=2)
        thresh_index = tf.where(Cls_pro_coor >= threshold_p)
        boundingbox = thresh_index
        # index_thre = tf.transpose(, perm=[1,0])
        bb1_tmp = (stride * tf.cast(boundingbox, tf.float32))*scale
        bb2_tmp = (stride * tf.cast(boundingbox, tf.float32) + 11.0)*scale
        bb1 = self._tffix(bb1_tmp)
        bb2 = self._tffix(bb2_tmp)
        boundingbox = tf.concat([bb1, bb2], 1)
        # index_0 = tf.gather(index_thre, 0)
        # index_1 = tf.gather(index_thre, 1)
        # i = tf.constant(0, dtype=tf.int32)
        # pick_index = tf.Variable([])
        # def cond(i, index_0, index_1, pick_index):
        #     return tf.greater(i, tf.size(index_0))
        # def body(i, index_0, index_1, pick_index):
        #     a_i = index_0[i]
        #     b_i = index_1[i]
        #     pick_index = tf.concat([pick_index, [a_i, b_i]], axis=0)
        #     i = i + 1
        #     return i, index_0, index_1, pick_index
        # re_index = tf.while_loop(cond, body, [i, index_0, index_1, pick_index],
        #                         shape_invariants=[i.get_shape(), index_0.get_shape(), index_1.get_shape(), tf.TensorShape([None, 2])])
        dx1 = tf.gather_nd(tf.gather(Roi, 0), thresh_index)
        dx2 = tf.gather_nd(tf.gather(Roi, 1), thresh_index)
        dx3 = tf.gather_nd(tf.gather(Roi, 2), thresh_index)
        dx4 = tf.gather_nd(tf.gather(Roi, 3), thresh_index)

        Cls_pro_GA = tf.gather(Cls_pro, 0, axis=2)
        score = tf.expand_dims(tf.gather_nd(Cls_pro_GA, thresh_index), axis=1)
        offset = tf.stack([dx1, dx2, dx3, dx4], axis=1)
        boundingbox = boundingbox + offset * 12.0*scale
        rectangles = tf.concat([boundingbox, score], axis=1)
        rectangles = self.rec2square(rectangles)
        return rectangles
    def pick_rect(self, rect_i, height_raw, width_raw):
        x1 = tf.maximum(0.0, tf.gather(rect_i, 0))
        y1 = tf.maximum(0.0, tf.gather(rect_i, 1))
        x2 = tf.minimum(tf.gather(width_raw, 0),  tf.gather(rect_i, 2))
        y2 = tf.minimum(tf.gather(height_raw, 0),  tf.gather(rect_i, 3))
        sc = tf.gather(rect_i, 4)
        def f1(): return x1, y1, x2, y2, sc
        def f2xy(): return tf.minimum(x1, x2), tf.minimum(y1, y2), tf.maximum(x1, x2), tf.maximum(y1, y2), sc
        cond_x = tf.greater(x1, x2)
        cond_y = tf.greater(y1, y2)
        cond_xy = cond_x | cond_y
        final = tf.cond(cond_xy, true_fn=f2xy, false_fn=f1)
        return final
    def _pick_chs(self, rect, height_raw, width_raw):
        x1, y1, x2, y2, sc = tf.map_fn(lambda x: self.pick_rect(x, height_raw, width_raw), rect, dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        return tf.stack([x1, y1, x2, y2, sc], axis=1)
    def _pick_chs_old(self, rect, img_shape):
        height_raw = tf.cast(img_shape[0], tf.float32)
        width_raw = tf.cast(img_shape[1], tf.float32)
        x1, y1, x2, y2, sc = tf.map_fn(lambda x: self.pick_rect(x, height_raw, width_raw), rect, dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        return tf.stack([x1, y1, x2, y2, sc], axis=1)
    def _NMS_own(self, rectangles, threshold, type):
        len_rect = tf.size(rectangles)
        condlen = tf.equal(len_rect, 0)
        def len0(): return rectangles
        def lennot0():
            x1 = tf.gather(rectangles, 0, axis=1)
            y1 = tf.gather(rectangles, 1, axis=1)
            x2 = tf.gather(rectangles, 2, axis=1)
            y2 = tf.gather(rectangles, 3, axis=1)
            s = tf.gather(rectangles, 4, axis=1)
            area = tf.multiply(x2 - x1 + 1, y2 - y1 + 1)
            I = tf.argsort(s)
            pick = tf.Variable([], dtype=tf.int32)
            def _cond(I, pick):
                return tf.not_equal(tf.size(I), 0)
            def _body(I, pick):
                I_size = tf.size(I)
                index_1 = tf.slice(I, [I_size - 1], [1])
                index_2 = tf.slice(I, [0], [I_size - 1])
                xx1 = tf.maximum(tf.gather(x1, index_1), tf.gather(x1, index_2))
                yy1 = tf.maximum(tf.gather(y1, index_1), tf.gather(y1, index_2))
                xx2 = tf.minimum(tf.gather(x2, index_1), tf.gather(x2, index_2))
                yy2 = tf.minimum(tf.gather(y2, index_1), tf.gather(y2, index_2))
                w = tf.maximum(0.0, xx2 - xx1 + 1.0)
                h = tf.maximum(0.0, yy2 - yy1 + 1.0)
                inter = w * h
                def istypeiom():
                    return inter / (tf.minimum(tf.gather(area, index_1), tf.gather(area, index_2)))
                def typenoetiom():
                    return inter / (tf.gather(area, index_1) + tf.gather(area, index_2) - inter)

                o = tf.cond(tf.equal(type, 'iom'), istypeiom, typenoetiom)

                pick = tf.concat([pick, index_1], axis=0)
                I_index_tmp = tf.where(o <= threshold)
                I_index = tf.gather(I_index_tmp, 0, axis=1)
                I = tf.gather(I, I_index, axis=0)
                return I, pick
            pick = tf.while_loop(_cond, _body, [I, pick], shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])])
            _pick = pick[1]
            result_rectangle = tf.gather(rectangles, _pick, axis=0)
            return result_rectangle
        result_rectangle = tf.cond(condlen, len0, lennot0)

        return result_rectangle
    def _NMS(self, rectangles, iou_threshold, max_output_size):
        boxes = tf.gather(rectangles, [0,1,2,3], axis=1)
        scores = tf.gather(rectangles, 4, axis=1)
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size, iou_threshold)
        selected = tf.gather(rectangles, selected_indices)
        return selected
    def post_data_Pnet(self, classifier, bbox_regress, scale, height_raw, width_raw):
        Cls_pro = tf.transpose(tf.gather(classifier, [1], axis=2), perm=[1, 0, 2])  # w, h
        Roi = tf.transpose(bbox_regress, perm=[2, 1, 0])                            # w, h
        rectangles = self._cls_bb2rect(Cls_pro, Roi, 1.0/scale)
        pick = self._pick_chs(rectangles, height_raw, width_raw)
        Pnet_rect = self._NMS_own(pick, 0.3, 'iou')
        # Pnet_rect = self._NMS(pick, 0.3, 100)
        return Pnet_rect, Pnet_rect, Pnet_rect, Pnet_rect, Pnet_rect
    def call(self, inputs):
        classifier = inputs[0]
        bbox_regress = inputs[1]
        scale = inputs[2]
        height_raw = inputs[3]
        width_raw = inputs[4]
        batch_data = (classifier, bbox_regress, scale, height_raw, width_raw)
        rects, _, _, _, _ = tf.map_fn(lambda x: self.post_data_Pnet(x[0], x[1], x[2], x[3], x[4]), batch_data, dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        return [rects]
class NMS_4_Pnetouts(Layer):
    def __init__(self, **kwargs):
        super(NMS_4_Pnetouts, self).__init__(**kwargs)
    def _crop(self, rects, img):
        rects_int = tf.cast(rects, tf.int32)
        def crop_image(img, crop):
            img_crop = tf.slice(img, [crop[1], crop[0], 0], [crop[3] - crop[1] +1, crop[2] - crop[0]+1, 3] )
            # img_crop_gather = tf.gather_nd()
            img_crop = tf.image.resize_images(img_crop, (24, 24))
            return img_crop
        cropped_image = tf.map_fn(lambda x: crop_image(img, x), elems=rects_int, dtype=(tf.float32))
        return cropped_image
    def call(self, inputs):
        rects = inputs[0]
        img = inputs[1]
        bb = tf.map_fn(lambda x: Pnet_post()._NMS_own(x, 0.7, 'iou'), rects, dtype=(tf.float32))
        data_batch = (bb, img)
        Pnet_4_Rnet = tf.map_fn(lambda x: self._crop(x[0], x[1]), data_batch, dtype=(tf.float32))
        return [Pnet_4_Rnet, bb]
class Rnet_out(Layer):
    def __init__(self, **kwargs):
        super(Rnet_out, self).__init__(**kwargs)
        self.r_net = _Rnet()
    def r_net_predict(self, xx, x):
        cls_R, bbox_R = self.r_net(xx)
        # outs = tf.concat([cls_R, bbox_R], 1)
        return cls_R, bbox_R  # outs[0:1] : cls; outs[2:-1] : bbox
    def call(self, inputs):
        batch_data = (inputs[0], inputs[1])
        outs_1, outs_2 = tf.map_fn(lambda x: self.r_net_predict(x[0], x[1]), batch_data, dtype=(tf.float32, tf.float32))
        return [outs_1, outs_2]
class out_post_Rnet(Layer):
    def __init__(self, **kwargs):
        super(out_post_Rnet, self).__init__(**kwargs)
        self.threshold = 0.6
    def crop_image_Rnet(self, crop, img):
        crop = tf.cast(crop, tf.int32)
        img_crop = tf.slice(img, [crop[1]-1, crop[0]-1, 0], [crop[3] - crop[1], crop[2] - crop[0], 3] )
        # img_crop_gather = tf.gather_nd()
        img_crop = tf.image.resize_images(img_crop, (48, 48))
        return img_crop
    def filter_face_Rnet(self, cls_pro, rio_pro, rectangles, img_raw, origin_h, origin_w):
        pick_index = tf.gather(tf.where(cls_pro >= self.threshold), 0, axis=1)
        x1 = tf.gather(tf.gather(rectangles, 0, axis=1), pick_index)
        y1 = tf.gather(tf.gather(rectangles, 1, axis=1), pick_index)
        x2 = tf.gather(tf.gather(rectangles, 2, axis=1), pick_index)
        y2 = tf.gather(tf.gather(rectangles, 3, axis=1), pick_index)
        sc = tf.gather(cls_pro, pick_index)
        dx1 = tf.gather(tf.gather(rio_pro, 0, axis=1), pick_index)
        dx2 = tf.gather(tf.gather(rio_pro, 1, axis=1), pick_index)
        dx3 = tf.gather(tf.gather(rio_pro, 2, axis=1), pick_index)
        dx4 = tf.gather(tf.gather(rio_pro, 3, axis=1), pick_index)
        w = x2 - x1
        h = y2 - y1
        x1 = x1 + dx1*w
        y1 = y1 + dx2*h
        x2 = x2 + dx3*w
        y2 = y2 + dx4*h
        rectangles = tf.stack([x1, y1, x2, y2, sc], axis=1)
        rectangles = Pnet_post().rec2square(rectangles)
        rectangles = Pnet_post()._NMS_own(rectangles, 0.3, 'iou')
        O_inputs = tf.map_fn(lambda x: self.crop_image_Rnet(x,img_raw), rectangles, dtype=(tf.float32))
        return O_inputs, O_inputs, O_inputs, rectangles
    def call(self, inputs):
        cls_pro = tf.gather(inputs[0], 1, axis=2)
        roi_pro = inputs[1]
        rectangles = inputs[2]
        origin_h = inputs[3]
        origin_w = inputs[4]
        img_raw = inputs[5]
        data_batch = (cls_pro, roi_pro, rectangles, img_raw)
        outs, _, _, rectangles = tf.map_fn(lambda x: self.filter_face_Rnet(x[0], x[1], x[2], x[3], origin_h, origin_w), data_batch, dtype=(tf.float32, tf.float32, tf.float32, tf.float32))
        cls_O, roi_O, pts_O = Onet_out()([outs])
        return [cls_O, roi_O, pts_O, rectangles]

class Onet_out(Layer):
    def __init__(self, **kwargs):
        self.output_dim = 3
        super(Onet_out, self).__init__(**kwargs)
        self.o_net = _Onet()
    def o_net_predict(self, xx):
        cls_O, roi_O, pts_O = self.o_net(xx)
        # outs = tf.concat([cls_R, bbox_R], 1)
        return cls_O, roi_O, pts_O
    def call(self, inputs):
        # cls_O, roi_O, pts_O  = self.o_net_predict(inputs)
        # batch_data = (inputs)
        cls_O, roi_O, pts_O = tf.map_fn(lambda x: self.o_net_predict(x), inputs, dtype=(tf.float32, tf.float32, tf.float32))
        return [cls_O, roi_O, pts_O]
    def compute_output_shape(self, input_shape):
        shape_a = input_shape[0]
        return [(shape_a, 2), (shape_a, 4), (shape_a, 10)]
class Onet_post_0(Layer):
    def __init__(self, **kwargs):
        self.threshold = 0.7
        super(Onet_post_0, self).__init__(**kwargs)

    def filter_face_48net(self,cls_prob,roi,pts,rectangles,height_raw, width_raw):
        prob = tf.gather(cls_prob, 1, axis=1)
        pick = tf.where(prob >= self.threshold)
        x1 = tf.gather(tf.gather(rectangles, 0, axis=1), pick)
        y1 = tf.gather(tf.gather(rectangles, 1, axis=1), pick)
        x2 = tf.gather(tf.gather(rectangles, 2, axis=1), pick)
        y2 = tf.gather(tf.gather(rectangles, 3, axis=1), pick)
        sc = tf.gather(prob, pick)
        dx1 = tf.gather(tf.gather(roi, 0, axis=1), pick)
        dx2 = tf.gather(tf.gather(roi, 1, axis=1), pick)
        dx3 = tf.gather(tf.gather(roi, 2, axis=1), pick)
        dx4 = tf.gather(tf.gather(roi, 3, axis=1), pick)
        w   = x2-x1
        h   = y2-y1
        pts0 = tf.multiply(w, tf.gather(tf.gather(pts,0, axis=1), pick))+ x1
        pts1 = tf.multiply(h, tf.gather(tf.gather(pts,5, axis=1), pick)) + y1
        pts2 = tf.multiply(w, tf.gather(tf.gather(pts,1, axis=1), pick)) + x1
        pts3 = tf.multiply(h, tf.gather(tf.gather(pts,6, axis=1), pick)) + y1
        pts4 = tf.multiply(w, tf.gather(tf.gather(pts,2, axis=1), pick)) + x1
        pts5 = tf.multiply(h, tf.gather(tf.gather(pts,7, axis=1), pick)) + y1
        pts6 = tf.multiply(w, tf.gather(tf.gather(pts,3, axis=1), pick)) + x1
        pts7 = tf.multiply(h, tf.gather(tf.gather(pts,8, axis=1), pick)) + y1
        pts8 = tf.multiply(w, tf.gather(tf.gather(pts,4, axis=1), pick)) + x1
        pts9 = tf.multiply(h, tf.gather(tf.gather(pts,9, axis=1), pick)) + y1

        x1 = x1 + tf.multiply(w, dx1)
        y1 = y1 + tf.multiply(h, dx2)
        x2 = x2 + tf.multiply(w, dx3)
        y2 = y2 + tf.multiply(h, dx4)
        rects = tf.concat([x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9], axis=1)
        # # pick = tf.Variable(tf.zeros([1, 15], dtype=tf.float32))
        # # pick = tf.TensorArray(dtype=tf.float32, size=15, dynamic_size=True, clear_after_read=False)
        # # pick = tf.placeholder(tf.float32, (None, 15))
        # # pick = tf.Variable([], dtype=tf.float32)
        pick = tf.stack([[rects[0, 0], rects[0, 1], rects[0, 2], rects[0,3], rects[0, 4],
                          rects[0,5],  rects[0,6],  rects[0,7],  rects[0,8],
                          rects[0,9],  rects[0,10], rects[0,11], rects[0,12],
                          rects[0,13], rects[0,14]]])
        # pick = tf.stack([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        i = tf.constant(0)
        row_rect = tf.shape(rects)[0]
        loop_cond = lambda i, rects, pick, width_raw, height_raw: i < row_rect
        def loop_body(i, rects, pick, width_raw,height_raw):
            x1 = tf.maximum(0.0, tf.gather_nd(rects, [i, 0]))
            y1 = tf.maximum(0.0, tf.gather_nd(rects, [i, 1]))
            x2 = tf.minimum(width_raw[0], tf.gather_nd(rects, [i, 2]))
            y2 = tf.minimum(height_raw[0], tf.gather_nd(rects, [i, 3]))
            tf_if_cond = tf.greater(x2, x1) & tf.greater(y2, y1)
            # def tf_true(x1, y1, x2, y2):
            #     return x1, y1, x2, y2
            # def tf_false(x1, y1, x2, y2):
            #     return tf.minimum(x1, x2), tf.minimum(y1, y2), tf.maximum(x1, x2), tf.maximum(y1, y2)
            # x1, y1, x2, y2 = tf.cond(tf_if_cond, lambda: tf_true(x1, y1, x2, y2), lambda: tf_false(x1, y1, x2, y2))
            # def tf_true():
            #     return x1, y1, x2, y2
            # def tf_false():
            #     return tf.minimum(x1, x2), tf.minimum(y1, y2), tf.maximum(x1, x2), tf.maximum(y1, y2)

            def f1(): return x1, y1, x2, y2
            def f2(): return tf.minimum(x1, x2), tf.minimum(y1, y2), tf.maximum(x1, x2), tf.maximum(y1, y2)

            x1, y1, x2, y2 = tf.cond(tf_if_cond, f1, f2)

            # stack_all_ = tf.stack([[x1_, y1_, x2_, y2_, rects[i, 4],rects[i,5], rects[i,6], rects[i,7],
            #                          rects[i,8],rects[i,9], rects[i,10], rects[i,11], rects[i,12],
            #                          rects[i,13], rects[i,14]]])
            stack_all = tf.stack([[x1, y1, x2, y2, rects[i, 4],rects[i,5], rects[i,6], rects[i,7],
                                     rects[i,8],rects[i,9], rects[i,10], rects[i,11], rects[i,12],
                                     rects[i,13], rects[i,14]]])
            # pick = tf.concat([pick, stack_all_], axis=0)
            pick = tf.concat([pick, stack_all], axis=0)
            # pick = tf.concat([pick, [x1, y1, x2, y2, rectangles[i,4],
            #                          rectangles[i,5], rectangles[i,6], rectangles[i,7], rectangles[i,8],
            #                          rectangles[i,9], rectangles[i,10], rectangles[i,11], rectangles[i,12],
            #                          rectangles[i,13], rectangles[i,14]]], axis=0)
            i = tf.add(i, 1)
            return i, rects, pick, width_raw, height_raw

        i, rectangles, pick, wid, heig = tf.while_loop(loop_cond, loop_body, [i, rects, pick, width_raw, height_raw],
                                   shape_invariants=[i.get_shape(), rects.get_shape(), tf.TensorShape([None, 15]),
                                                     width_raw.get_shape(), height_raw.get_shape()])
        pick = tf.slice(pick, [1,0], [-1, -1])
        pick_NMS = tf.gather(pick, [0, 1, 2, 3, 4], axis=1)
        rectangles = Pnet_post()._NMS_own(pick_NMS, 0.3, 'iom')

        return i, rectangles, pick, wid, heig, pick
    def call(self, inputs):
        cls = inputs[0]
        roi = inputs[1]
        pts = inputs[2]
        rect = inputs[3]
        origin_h = inputs[4]
        origin_w = inputs[5]
        batch_data = (cls, roi, pts, rect, origin_h, origin_w)
        # i, rectangles, pick, wid, heig, pick = tf.map_fn(lambda x: self.filter_face_48net(x[0], x[1], x[2], x[3], x[4], x[5]), batch_data,
        #                                                                                            dtype=(tf.float32, tf.float32))
        # , tf.float32, tf.float32, tf.float32, tf.float32,
        #                                                                                               tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
        #                                                                                               tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        i, rectangles, pick, wid, heig, pick_1 = tf.map_fn(lambda x: self.filter_face_48net(x[0], x[1], x[2], x[3], x[4], x[5]), batch_data,
                                        dtype=(tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

        return [pick]

    def compute_output_shape(self, input_shape):
        # shape_a = input_shape[0][0]
        return [(None, None, 15)]
            # , (None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1),
            #     (None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1),
            #     (None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1), (None, None, 1)]

def mainmodel():
    img = Input(shape=[None, None, 3])
    img_0 = Input(shape=[None, None, 3])
    scale_0 = Input(shape=(1,))
    img_1 = Input(shape=[None, None, 3])
    scale_1 = Input(shape=(1,))
    origin_h = Input(shape=(1,))
    origin_w = Input(shape=(1,))
    ################## --- Pnet -----############
    outs_0 = _Pnet()([img_0, scale_0, origin_h, origin_w])
    outs_1 = _Pnet()([img_1, scale_1, origin_h, origin_w])
    outs = concatenate([outs_0, outs_1], axis=1)
    out_nms, rectangles_p = NMS_4_Pnetouts()([outs, img])
    out_Rnet_1, out_Rnet_2 = Rnet_out()([out_nms, out_nms])
    cls_O, roi_O, pts_O, rectangles_r = out_post_Rnet()([out_Rnet_1, out_Rnet_2, rectangles_p, origin_h, origin_w, img])
    rects = Onet_post_0()([cls_O, roi_O, pts_O, rectangles_r, origin_h, origin_w])
    model = Model(inputs=[img, img_0, scale_0, img_1, scale_1, origin_h, origin_w], outputs=[rects])
    # Save the weights
    # model.save_weights('MTCNNallNet_weights.h5')
    model.save('MTCNNallNet_weights0.h5')
    # Save the model architecture
    with open('MTCNNallNet_architecture.json', 'w') as f:
        f.write(model.to_json())

    return model


def main():

    print('Load picture again!')
    img_ = cv2.imread(r'F:\TEST\2.jpg')
    img = (img_ - 127.5) / 127.5
    img_raw = np.expand_dims(img, axis=0)
    origin_h, origin_w, ch = img.shape

    # scales = [1, 0.6797385]
    scales = [0.09587285, 0.06797385]

    scale_0 = scales[0]
    hs_0 = int(origin_h * scale_0)
    ws_0 = int(origin_w * scale_0)
    scale_img_in_0 = np.expand_dims(cv2.resize(img, (ws_0, hs_0)), axis=0)
    scale_in_0 = np.expand_dims(scale_0, axis=0)

    scale_1 = scales[1]
    hs_1 = int(origin_h * scale_1)
    ws_1 = int(origin_w * scale_1)
    scale_img_in_1 = np.expand_dims(cv2.resize(img, (ws_1, hs_1)), axis=0)
    scale_in_1 = np.expand_dims(scale_1, axis=0)

    height_in = np.expand_dims(origin_h, axis=0)
    width_in = np.expand_dims(origin_w, axis=0)

    # model.summary()
    outs = mainmodel().predict([img_raw, scale_img_in_0, scale_in_0, scale_img_in_1, scale_in_1, height_in, width_in])
    model5 = load_model('MTCNNallNet_weights0.h5', custom_objects={'Pnet_post':Pnet_post,
                                                                  'NMS_4_Pnetouts':NMS_4_Pnetouts,
                                                                  'Rnet_out':Rnet_out,
                                                                  'out_post_Rnet':out_post_Rnet,
                                                                  'Onet_post_0':Onet_post_0})
    outs_2 = model5.predict([img_raw, scale_img_in_0, scale_in_0, scale_img_in_1, scale_in_1, height_in, width_in])
    rect = outs[0][:,0:5]
    bb_num = np.shape(rect)[0]
    img = img_

    colors = 255*(np.random.standard_normal(size=(bb_num, 3)) + 1)/2
    for i in range(bb_num):
        bb = rect[i, 0:4]
        color = colors[i, :]
        # img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (color[0], color[1], color[2]), 5)
        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 5)
        img = cv2.putText(img, '#{}:{}'.format(i,rect[i, 4]), (int((bb[2]+bb[1])/2), bb[3]), fontFace=4, fontScale=0.5, color=(255,0,0))
        cv2.imshow('img', img)

if __name__ == '__main__':
    main()


