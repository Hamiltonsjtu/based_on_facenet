# _*_ coding: utf-8 _*_
import sys
import argparse
import os
import random
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.applications.inception_v3 import InceptionV3
# from tensorflow.python.keras.layers import GlobalAveragePooling2D,Dropout,Dense,Input
# from tensorflow.python.keras.models import Model,load_model
# from tensorflow.python.keras.optimizers import SGD, Adam, Adagrad

from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, RepeatVector, Reshape, Concatenate
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
# from tensorflow.python.keras.utils import Sequence
import tensorflow as tf


def sta_sample(datapath,bags):
    for b in bags:
        if b == "yes":
            true_path = os.path.join(datapath, b)
            imgs = os.listdir(true_path)
            yes_num = len(imgs)
        elif b == "no":
            false_path = os.path.join(datapath, b)
            imgs = os.listdir(false_path)
            no_num = len(imgs)
        else:
            unsure_path = os.path.join(datapath, b)
            imgs = os.listdir(unsure_path)
            nosure_num = len(imgs)
    return yes_num, no_num, nosure_num

# from tensorflow.python.keras.callbacks import ModelCheckpoint
def getdataset(datapath):
    bags = os.listdir(datapath)
    imgs_path = []
    imgs_y = []

    yes_num, no_num, nosure_num = sta_sample(bags)
    sample_num = np.minimum(yes_num+no_num, nosure_num)

    for b in bags:
        if b == "yes":
            true_path = os.path.join(datapath, b)
            imgs = os.listdir(true_path)
            for img in imgs:
                img_path = os.path.join(true_path, img)
                if img_path[-4:] == '.jpg' or img_path[-4:] == '.png':
                    imgs_path.append(img_path)
                    imgs_y.append(1)
        elif b == "no":
            false_path = os.path.join(datapath, b)
            imgs = os.listdir(false_path)
            for i in imgs:
                img_path = os.path.join(false_path, i)
                if img_path[-4:] == '.jpg' or img_path[-4:] == '.png':
                    imgs_path.append(img_path)
                    imgs_y.append(0)
        else:
            unsure_path = os.path.join(datapath, b)
            imgs = os.listdir(unsure_path)
            random.seed(666)
            random.shuffle(imgs)
            imgs_sl = imgs[0:sample_num]
            for i in imgs_sl:
                img_path = os.path.join(unsure_path, i)
                if img_path[-4:] == '.jpg' or img_path[-4:] == '.png':
                    imgs_path.append(img_path)
                    imgs_y.append(2)
    assert len(imgs_path) == len(imgs_y)
    randnum = random.randint(0, len(imgs_y))
    random.seed(randnum)
    random.shuffle(imgs_path)
    random.seed(randnum)
    random.shuffle(imgs_y)
    # for i in range(len(imgs_y)):
    #     print(imgs_path[i],imgs_y[i])
    return imgs_path, imgs_y


def read_pic(path, IMAGE_DIMS):
    try:
        img = cv2.imread(path)  # target_size参数前面是高
        x1 = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]), cv2.INTER_AREA)
        x1 = x1[:, :, ::-1]
    except Exception as e:
        print('=======  CV READ IMAGE EXCEPTION  ========')
        x1 = None
    return x1


def define_model(IMAGE_DIMS, weight_dir):
    x1 = Input(shape=IMAGE_DIMS)
    head = Input(shape=(256,))
    screen = Input(shape=(256,))
    print('-----------START LOAD INCEPTION MODEL------------')
    base_model = InceptionV3(weights=weight_dir, include_top=False)
    poolings = GlobalAveragePooling2D()
    print('------base_model_layers', len(base_model.layers))
    base_model.summary()
    # 增加新的输出层
    # x = base_model.get_layer('mixed10').output
    x1 = base_model(x1)
    # print(x.shape)
    # print(x.name)
    x1 = poolings(x1)  # 全局平均池化层
    # print(x.shape)
    # x = Dropout(0.5)(x)
    x_img = Dense(512, activation='relu', name='full_connect')(x1)
    head = RepeatVector(head, 64)
    screen = RepeatVector(screen, 64)
    x_h = Reshape(-1)(head)
    x_s = Reshape(-1)(screen)
    x = Concatenate([x_img, x_h, x_s])
    # predictions = Dense(len(CLASSES), activation='sigmoid')(x)  # 输出，mlb(6)个类
    predictions = Dense(3, activation='softmax', name='two_label')(x)  # 输出，mlb(6)个类

    model = Model(inputs=x1, outputs=predictions)
    return model


def setup_to_fine_tune(model, base_model, INIT_LR, EPOCHS):
    '''GAP_LAYER = 80  # 需要根据实际效果进行确定'''
    trainable = False
    for i, layer in enumerate(base_model.layers):
        layer.trainable = trainable
        if layer.name == 'mixed7':
            trainable = True
    # model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['mse'])
    # model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.compile(optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), loss='binary_crossentropy', metrics=['accuracy'])


class Sequ_pic(Sequence):
    def __init__(self, x, y, anno_path, batch_size, IMAGE_DIMS):
        self.x = x
        self.y = y
        self.anno_path = anno_path
        self.batch_size = batch_size
        self.IMAGE_DIMS = IMAGE_DIMS

    def __getitem__(self, idx):
        train_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        train_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # print(train_y)
        X1 = []
        XH = []
        XS = []
        Y = []
        for i in range(len(train_x)):
            x1 = self.read_pic(train_x[i], self.IMAGE_DIMS)
            if x1 is None:
                continue
            X1.append(x1)
            XH.append(self.read_pic_adjust(x1, train_x[i], self.anno_path)[0])
            XS.append(self.read_pic_adjust(x1, train_x[i], self.anno_path)[1])
            temp = np.zeros(3)
            temp[train_y[i]] = 1
            Y.append(temp)

            X1 = np.array(X1, dtype="uint8")
            XH = np.array(XH, dtype="uint8")
            XS = np.array(XS, dtype="uint8")
            Y = np.array(Y)
        return [X1, XH, XS], Y

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def point_relocate(self, image, ori_w, ori_h, w, h, point_x, point_y):
        w_ratio = float(w) / float(ori_w)
        h_ratio = float(h) / float(ori_h)
        ratio = np.minimum(w_ratio, h_ratio)
        w_tmp = int(ratio * ori_w)
        h_tmp = int(ratio * ori_h)
        image = cv2.resize(image, (w_tmp, h_tmp))
        point_x = int(ratio * point_x)
        point_y = int(ratio * point_y)
        top = h - h_tmp
        right = w - w_tmp
        bottom = 0
        left = 0
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img, point_x, point_y

    def read_pic(self, path, IMAGE_DIMS):
        try:
            img = cv2.imread(path)  # target_size参数前面是高
            x1 = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]), cv2.INTER_AREA)
            x1 = x1[:, :, ::-1]
        except Exception as e:
            print('=======  CV READ IMAGE EXCEPTION  ========')
            x1 = None

        return x1

    def read_pic_adjust(self, image, image_path, anno_path):
        ori_h, ori_w, c = image.shape
        anno = pd.read_table(anno_path)
        ## we recommend the ratio of width/height is 0.5625,
        # namely the ratio of 720*1280 or 1080*1920
        w = 720
        h = 1280
        df = anno[anno.name == os.path.split(image_path)[1]].iloc[0]
        head_xtl = df['head_xtl']
        head_ytl = df['head_ytl']
        head_xbr = df['head_xbr']
        head_ybr = df['head_ybr']

        screen_xtl = df['screen_xtl']
        screen_ytl = df['screen_ytl']
        screen_xbr = df['screen_xbr']
        screen_ybr = df['screen_ybr']

        img_n, head_xbr_u, head_ybr_u = self.point_relocate(image, ori_w, ori_h, w, h, head_xbr, head_ybr)
        _, head_xtl_u, head_ytl_u = self.point_relocate(image, ori_w, ori_h, w, h, head_xtl, head_ytl)
        _, screen_xbr_u, screen_ybr_u = self.point_relocate(image, ori_w, ori_h, w, h, screen_xbr, screen_ybr)
        _, screen_xtl_u, screen_ytl_u = self.point_relocate(image, ori_w, ori_h, w, h, screen_xtl, screen_ytl)

        head = [head_xtl_u, head_ybr_u, head_xbr_u, head_ytl_u]
        screen = [screen_xtl_u, screen_ybr_u, screen_xbr_u, screen_ytl_u]

        return head, screen



def main(args):
    EPOCHS = 200
    INIT_LR = 1e-4
    BS = 32
    WORKERS = 10
    IMAGE_DIMS = (720, 1280, 3)
    datapath = args.dataset
    weight_dir = args.weight_dir
    anno_path = args.anno_path
    imgs_path, imgs_y = getdataset(datapath)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.Session(config=config)
    filepath = 'FocusScreenModel'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = filepath + '/' + '{epoch:03d}.h5'
    X_train, X_test, y_train, y_test = train_test_split(imgs_path, imgs_y, random_state=2)
    print(len(X_train), len(X_test), sum(y_test))
    pa_model = define_model(IMAGE_DIMS, weight_dir)

    encoder = pa_model.get_layer('inception_v3')
    setup_to_fine_tune(pa_model, encoder, INIT_LR, EPOCHS)
    pa_model.summary()

    train_sequence = Sequ_pic(X_train, y_train, anno_path, BS, IMAGE_DIMS)
    test_sequence = Sequ_pic(X_test, y_test, anno_path, BS, IMAGE_DIMS)
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    H = pa_model.fit_generator(
        train_sequence,
        steps_per_epoch=len(X_train) // BS,
        validation_data=test_sequence, validation_steps=len(X_test) // BS, epochs=EPOCHS,
        workers=WORKERS, use_multiprocessing=True, verbose=1, callbacks=[checkpoint])


def parse_arguments(argv):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=r"F:\FocusScreen\FocusImage",
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-a", "--anno_path", default=r"F:\FocusScreen\eyeposition.txt",
                    help="path to annotation dataset (i.e., directory of annotation file)")
    ap.add_argument("-v", "--vec_len", type=int, default=512,
                    help="length of feature vector")
    ap.add_argument('--weight_dir', type=str,
                    help='Weights restored from pre-trained InceptionV3.',
                    default='./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    ap.add_argument("--modeltest", default="/home/Downloads/041.h5",
                    help="path to input test (i.e., directory of images)")
    return ap.parse_args(argv)


def acc(args):
    model_path = args.modeltest
    model = load_model(model_path)
    model.summary()
    datapath = args.dataset
    weight_dir = args.weight_dir
    imgs_path, imgs_y = getdataset(datapath)
    batch_size = 50
    IMAGE_DIMS = (300, 300, 3)
    step = int(np.ceil(len(imgs_path) / batch_size))
    print(step)
    real_y = []
    pre_all = []
    for i in range(step):
        sample_image = imgs_path[batch_size * i:batch_size * (i + 1)]
        sample_class = imgs_y[batch_size * i:batch_size * (i + 1)]
        read_imgs = []

        for j in range(len(sample_image)):
            temp = read_pic(sample_image[j], IMAGE_DIMS)
            if temp is not None:
                read_imgs.append(temp)
                real_y.append(sample_class[j])
        pic_list = np.array(read_imgs)
        pre_y = model.predict(pic_list)
        pre_all.extend(np.argmax(pre_y, 1))
        print(i)
        # print(pre_all)

    # print(np.argmax(pre_y,1))
    # print(real_y)
    print(np.mean(np.equal(pre_all, real_y)))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    # acc(parse_arguments(sys.argv[1:]))
