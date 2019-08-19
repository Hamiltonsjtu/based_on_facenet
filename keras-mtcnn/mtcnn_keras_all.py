import sys
import tools_matrix as tools
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import keras as KK
import keras.backend as K
from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute, Lambda, concatenate
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
from keras.layers import Layer, Concatenate, Reshape
def _Pnet(weight_path = './12net.h5'):
    input_img_0 = Input(shape=[None, None, 3])
    scale = Input(shape=())
    x_0 = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input_img_0)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU1')(x_0)
    x_0 = MaxPool2D(pool_size=2)(x_0)
    x_0 = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x_0)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU2')(x_0)
    x_0 = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x_0)
    x_0 = PReLU(shared_axes=[1,2],name='PReLU3')(x_0)
    classifier_0 = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x_0)
    bbox_regress_0 = Conv2D(4, (1, 1), name='conv4-2')(x_0)
    outs_0 = Pnet_post(name='Pnet_output_post')([classifier_0, bbox_regress_0, input_img_0, scale])
    model = Model(inputs=[input_img_0, scale], outputs=outs_0)
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

# class _Rnet_post(Layer):
#     def __init__(self, **kwargs):
#         super(_Rnet_post, self).__init__(**kwargs)
#         self.threshold = 0.7
#     def _post_data_Rnet(self, classifier, bbox_regress, input_rects, input_height, input_width):
#         return 1.0
#     def call(self,inputs):
#         classifier = inputs[0]
#         bbox_regress = inputs[1]
#         input_rects = inputs[2]
#         input_height = inputs[3]
#         input_width = inputs[4]
#         batch_data = (classifier, bbox_regress, input_rects)
#         bb1, bb2, bb3 = tf.map_fn(lambda x: self.post_data_Rnet(x[0], x[1], x[2], input_height, input_width), batch_data, dtype=(tf.float32, tf.float32, tf.float32))
#         return bb1, bb2, bb3
# def _Rnet(weight_path = 'model24.h5'):
#     input = Input(shape=[24, 24, 3]) # change this shape to [None,None,3] to enable arbitraty shape input
#     input_rects = Input(shape=[None, 4])
#     input_height = Input(shape=(1,))
#     input_width = Input(shape=(1,))
#     x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
#     x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
#     x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
#
#     x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
#     x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
#     x = MaxPool2D(pool_size=3, strides=2)(x)
#
#     x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
#     x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
#     x = Permute((3, 2, 1))(x)
#     x = Flatten()(x)
#     x = Dense(128, name='conv4')(x)
#     x = PReLU( name='prelu4')(x)
#     classifier = Dense(2, activation='softmax', name='conv5-1')(x)
#     bbox_regress = Dense(4, name='conv5-2')(x)
#
#     R_out = _Rnet_post([classifier, bbox_regress, input_rects, input_height, input_width])
#     model = Model([input], [classifier, bbox_regress, input_rects, input_height, input_width])
#     model.load_weights(weight_path, by_name=True)
#     return model
class Pnet_post(Layer):
    def __init__(self, **kwargs):
        super(Pnet_post, self).__init__(**kwargs)
        self.scale = 1.0
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
        rec_1 = tf.gather(rectangles,0, axis=-1) + w*0.5 - l*0.5
        rec_2 = tf.gather(rectangles,1, axis=-1) + h*0.5 - l*0.5
        rec_3 = rec_1 + l
        rec_4 = rec_2 + l
        rec_5 = tf.gather(rectangles,4, axis=-1)
        rectangles = tf.transpose(tf.concat([[rec_1], [rec_2], [rec_3], [rec_4], [rec_5]], axis=0), perm=[1,0])
        return rectangles
    def _cls_bb2rect(self, Cls_pro, Roi, scale):
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
        boundingbox = tf.cast(tf.slice(thresh_index, [0, 0], [-1, 2]), dtype=tf.float32)
        bb1_tmp = (stride * boundingbox)*scale
        bb2_tmp = (stride * boundingbox + 11.0)*scale
        bb1 = self._tffix(bb1_tmp)
        bb2 = self._tffix(bb2_tmp)
        boundingbox = tf.concat([bb1, bb2], -1)
        dx1 = tf.gather_nd(tf.gather(Roi, 0), [thresh_index])
        dx2 = tf.gather_nd(tf.gather(Roi, 1), [thresh_index])
        dx3 = tf.gather_nd(tf.gather(Roi, 2), [thresh_index])
        dx4 = tf.gather_nd(tf.gather(Roi, 3), [thresh_index])
        Cls_pro_GA = tf.gather(Cls_pro, [0], axis=2)
        score = tf.gather_nd(Cls_pro_GA, [thresh_index])
        offset = tf.stack([dx1, dx2, dx3, dx4], axis=-1)
        boundingbox = boundingbox + offset * 12.0*scale
        rectangles = tf.squeeze(tf.concat([boundingbox, score], axis=-1),[0])
        rectangles = self.rec2square(rectangles)
        return rectangles
    def pick_rect(self, rect_i, height_raw, width_raw):
        x1 = tf.maximum(0.0, rect_i[0])
        y1 = tf.maximum(0.0, rect_i[1])
        x2 = tf.minimum(width_raw, rect_i[2])
        y2 = tf.minimum(height_raw, rect_i[3])
        sc = rect_i[4]
        def f1(): return x1, y1, x2, y2, sc
        def f2x(): return x2, y1, x1, y2, sc
        def f2y(): return x1, y2, x2, y1, sc
        def f2xy(): return x2, y2, x1, y1, sc
        cond_x = tf.greater(x1, x2)
        cond_y = tf.greater(y1, y2)
        cond_xy = cond_x & cond_y
        final = tf.cond(cond_x, f2x, f1)
        final = tf.cond(cond_y, f2y, f1)
        final = tf.cond(cond_xy, f2xy, f1)
        return final
    def _pick_chs(self, rect, img):
        img_shape = tf.shape(img)
        height_raw = tf.cast(img_shape[0], tf.float32)
        width_raw = tf.cast(img_shape[1], tf.float32)
        x1, y1, x2, y2, sc = tf.map_fn(lambda x: self.pick_rect(x, height_raw, width_raw), rect, dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        return tf.stack([x1, y1, x2, y2, sc], axis=1)
    def _pick_chs_old(self, rect, img_shape):
        height_raw = tf.cast(img_shape[0], tf.float32)
        width_raw = tf.cast(img_shape[1], tf.float32)
        x1, y1, x2, y2, sc = tf.map_fn(lambda x: self.pick_rect(x, height_raw, width_raw), rect, dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        return tf.stack([x1, y1, x2, y2, sc], axis=1)
    def _NMS(self, rectangles, iou_threshold, max_output_size):
        boxes = tf.gather(rectangles, [0,1,2,3], axis=1)
        scores = tf.gather(rectangles, 4, axis=1)
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size, iou_threshold)
        selected = tf.gather(rectangles, selected_indices)
        return selected
    def _crop(self, img, rects):
        rects_int = tf.cast(rects, tf.int32)
        def crop_image(img, crop):
            img_crop = tf.slice(img, [crop[1], crop[0], 0], [crop[3] - crop[1] + 1, crop[2] - crop[0] + 1, 3] )
            img_crop = tf.image.resize_images(img_crop, (24, 24))
            return img_crop
    def post_data_Pnet(self, classifier, bbox_regress, img, scale):
        Cls_pro = tf.transpose(tf.gather(classifier, [1], axis=2), perm=[1, 0, 2])
        Roi = tf.transpose(tf.slice(bbox_regress, [0,0,0], [-1,-1,-1]), perm=[2, 1, 0])
        rectangles = self._cls_bb2rect(Cls_pro, Roi, 1.0/scale)
        pick = self._pick_chs(rectangles, img)
        Pnet_rect = self._NMS(pick, 0.3, 100)
        Pnet_4_Rnet = self._crop(img, Pnet_rect)
        return Pnet_rect, Pnet_rect, Pnet_rect
    def call(self, inputs):
        classifier = inputs[0]
        bbox_regress = inputs[1]
        img = inputs[2]
        scales = inputs[3]
        batch_data = (classifier, bbox_regress, img)
        bb, _, _ = tf.map_fn(lambda x: self.post_data_Pnet(x[0], x[1], x[2], scales), batch_data, dtype=(tf.float32, tf.float32, tf.float32))
        return [bb]
class NMS_4_Pnetouts(Layer):
    def __init__(self, **kwargs):
        super(NMS_4_Pnetouts, self).__init__(**kwargs)
    def _NMS(self, rect):
        bboxes = tf.gather(rect, [0,1,2,3], axis=-1)
        scores = tf.gather(rect, 4, axis=-1)
        selected_indices = tf.image.non_max_suppression(bboxes, scores, max_output_size=100, iou_threshold=0.7)
        selected = tf.gather(rect, selected_indices)
        return selected
    def _crop(self, rects, img):
        rects_int = tf.cast(rects, tf.int32)
        def crop_image(img, crop):
            img_crop = tf.slice(img, [crop[1]-1, crop[0]-1, 0], [crop[3] - crop[1], crop[2] - crop[0], 3] )
            # img_crop_gather = tf.gather_nd()
            img_crop = tf.image.resize_images(img_crop, (24, 24))
            return img_crop
        cropped_image = tf.map_fn(lambda x: crop_image(img, x), elems=rects_int, dtype=(tf.float32))
        return cropped_image
    def call(self, inputs):
        rects = inputs[0]
        img = inputs[1]
        bb = tf.map_fn(lambda x: self._NMS(x), rects, dtype=(tf.float32))
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
    def filter_face_Rnet(self, cls_pro, rio_pro, rectangles, origin_h, origin_w):
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
        rectangles = Pnet_post()._NMS(rectangles, 0.3, 100)
        return rectangles, rectangles, rectangles
    def call(self, inputs):
        cls_pro = tf.gather(inputs[0], 1, axis=2)
        roi_pro = inputs[1]
        rectangles = inputs[2]
        origin_h = inputs[3]
        origin_w = inputs[4]
        data_batch = (cls_pro, roi_pro, rectangles)
        outs, _, _ = tf.map_fn(lambda x: self.filter_face_Rnet(x[0], x[1], x[2], origin_h, origin_w), data_batch, dtype=(tf.float32, tf.float32, tf.float32))
        return outs


def mainmodel():
    #### -------------- P-net ------------------####
    img = Input(shape=[None, None, 3])

    # recommand scales = [1.5, 1.06, 0.75, 0.53, 0.38, 0.27]
    scale_0 = Lambda(lambda x: tf.constant([0.09587285]))(img)
    scale_1 = Lambda(lambda x: tf.constant([0.06797385]))(img)
    # scale_0 = Lambda(lambda x: tf.constant([1.0]))(img)
    # scale_1 = Lambda(lambda x: tf.constant([0.709]))(img)
    # scale_2 = Lambda(lambda x: tf.constant([0.5027]))(img)
    # scale_3 = Lambda(lambda x: tf.constant([0.3564]))(img)
    origin_h = Lambda(lambda x: tf.cast(tf.shape(x)[1], tf.float32))(img)
    origin_w = Lambda(lambda x: tf.cast(tf.shape(x)[2], tf.float32))(img)
    img_0 = Lambda(lambda x: tf.image.resize_images(x, tf.cast(scale_0*(origin_h, origin_w),tf.int32)))(img)
    img_1 = Lambda(lambda x: tf.image.resize_images(x, tf.cast(scale_1*(origin_h, origin_w),tf.int32)))(img)
    # img_2 = Lambda(lambda x: tf.image.resize_images(x, tf.cast(scale_2*(origin_h, origin_w),tf.int32)))(img)
    # img_3 = Lambda(lambda x: tf.image.resize_images(x, tf.cast(scale_3*(origin_h, origin_w),tf.int32)))(img)
    outs_0 = _Pnet()([img_0, scale_0])
    outs_1 = _Pnet()([img_1, scale_1])
    # outs_2 = _Pnet(Pnet_weight_path)([img_2, scale_2])
    # outs_3 = _Pnet(Pnet_weight_path)([img_3, scale_2])
    outs = concatenate([outs_0, outs_1], axis=1)
    #### -------------- R-net ------------------####
    out_nms, rectangles = NMS_4_Pnetouts()([outs, img])
    # out_Rnet_1, out_Rnet_2 = Rnet_out()([out_nms, out_nms])
    # out_postRnet = out_post_Rnet()([out_Rnet_1, out_Rnet_2, rectangles, origin_h, origin_w])
    model = Model(inputs=[img], outputs=[out_nms, rectangles])
    return model

# Pnet = _Pnet(r'12net.h5')
Network = mainmodel()
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
    img_0 = (img.copy() - 127.5) / 127.5
    # img_1 = cv2.resize(img_0, scales[1]*(origin_w, origin_h))
    # img_2 = cv2.resize(img_0, scales[2]*(origin_w, origin_h))
    img_0 = np.expand_dims(img_0, axis=0)
    # img_1 = np.expand_dims(img_1, axis=0)
    # img_2 = np.expand_dims(img_2, axis=0)
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
    # output = Pnet.predict([img_0, img_1, img_2])
    output = Network.predict(img_0)
    # cls_pro, roi = Pnet_out(ouput)
    # model = Model(ouput, pro_bbox)
    # image_scaled = model.predict(img)
    print('Done')


