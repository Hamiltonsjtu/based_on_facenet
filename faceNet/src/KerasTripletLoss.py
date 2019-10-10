#_*_ coding: utf-8 _*_
'''

'''
# import the necessary packages
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD,Adam,Adagrad
import pandas as pd
import re
import numpy as np
import argparse
import random
import cv2
import os
import sys
import time
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import math
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, AveragePooling2D, Flatten, Softmax, Subtract, Lambda, Reshape, Minimum, Concatenate
import tensorflow.python.keras.layers.normalization as normlize
from tensorflow.python.keras.models import Model,load_model
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.keras.layers.core import Dropout

from tensorflow.python.ops.linalg.linalg_impl import norm

def generate_arrays(path, batch_size,MARGIN,GAP,IMAGE_DIMS,class_lab):
    X1 = []
    X2 = []
    X3 = []
    Y  = []
    Y1 = []
    Y2 = []
    while 1:
        #random.seed(int(time.time()))
        # random.shuffle(path)
        total = len(path)
        cnt = 0
        for i,file in enumerate(path):
            if file[-4:] == '.png' or file[-4:] == '.jpg' :
                #print("F1", file)
                x1 = read_pic(file,IMAGE_DIMS)
                x2 = None
                x3 = None
                if x1 is None:
                    continue
                same_people=[j for j,v in enumerate(class_lab) if v==class_lab[i]]
                if len(same_people)<2:
                    continue
                np.random.shuffle(same_people)
                p_people=None
                for j in same_people:
                    if j==i:
                        continue
                    else:
                        p_people=j
                        break
                x2 = read_pic(path[p_people], IMAGE_DIMS)
                if x2 is None:
                    continue

                while True:
                    idx = random.randint(0,total-1)
                    if idx in same_people:
                        continue
                    file2 = path[idx]
                    if file2 is not file and (file2[-4:] == '.png' or file2[-4:] == '.jpg'):
                        #print("F2", file2)
                        x3 = read_pic(path[idx], IMAGE_DIMS)
                        if x3 is None:
                            continue
                        else:
                            break
                y1 = [MARGIN]
                y2 = [GAP]
                X1.append(x1)
                X2.append(x2)
                X3.append(x3)
                Y.append([MARGIN,GAP])
                # Y.append([MARGIN])
                Y1.append(y1)
                Y2.append(y2)
                cnt += 1

                if len(Y) == batch_size:
                    cnt = 0
                    X1 = np.array(X1, dtype="uint8")
                    X2 = np.array(X2, dtype="uint8")
                    X3 = np.array(X3, dtype="uint8")
                    Y  = np.array(Y)
                    Y1 = np.array(Y1)
                    Y2 = np.array(Y2)
                    #print("Feed one batch:", len(Y))
                    yield ([X1,X2,X3,Y1,Y2], Y)
                    X1 = []
                    X2 = []
                    X3 = []
                    Y  = []
                    Y1 = []
                    Y2 = []
            else:
                continue
#
class BaseModelCheckpoint(Callback):
    def __init__(self, filepath, base_model=None, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(BaseModelCheckpoint, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.base_model = self.base_model if self.base_model is not None else self.model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)
            fname=self.filepath+'_'+str(epoch)+'epoch.h5'
            fp=os.path.join(self.filepath,fname)
            filepath = fp.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

# ####### 构建基础模型
# ## 增加模型输入
#
# restore="triplet_model_latest.h5"
#
def max_error(y_true, y_pred):
    '''diff = K.abs(y_true - y_pred)
    K.max(Lambda(x:))'''
    return K.max(K.abs(y_true - y_pred))
#
def abs_pre(y_true, y_pred):
    return K.abs(y_pred)


def define_model(IMAGE_DIMS,VEC_LEN,weight_dir):
    i10 = Input(shape=IMAGE_DIMS)
    i20 = Input(shape=IMAGE_DIMS)
    i30 = Input(shape=IMAGE_DIMS)
    t1 = Input(shape=(1,))
    t2 = Input(shape=(1,))

    i1 = Lambda(lambda x: tf.div(tf.subtract(tf.cast(x,tf.float32), 127.5),127.5))(i10)
    i2 = Lambda(lambda x: tf.div(tf.subtract(tf.cast(x,tf.float32), 127.5),127.5))(i20)
    i3 = Lambda(lambda x: tf.div(tf.subtract(tf.cast(x,tf.float32), 127.5),127.5))(i30)
    # i1 = tf.cast(i1,tf.float32)
    # i2 = tf.cast(i2, tf.float32)
    # i3 = tf.cast(i3, tf.float32)
    #
    # i1=tf.div(tf.subtract(i1,127.5),127.5)
    # i2 = tf.div(tf.subtract(i2, 127.5), 127.5)
    # i3 = tf.div(tf.subtract(i3, 127.5), 127.5)
    print("[INFO] Weights restored from pre-trained InceptionV3!")
    encoder = InceptionV3(weights=weight_dir, include_top=False)
    pooling = GlobalAveragePooling2D()
    def l2_normalize(x):
        return K.expand_dims(K.l2_normalize(x, 1))
    val = Lambda(l2_normalize, name='margin')()
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                    beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                    moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                    beta_constraint=None, gamma_constraint=None)

    output = Dense(VEC_LEN, activation='tanh', name='encoder_output')

    o1 = encoder(i1)
    o2 = encoder(i2)
    o3 = encoder(i3)
    # o1 = i1
    # o2 = i2
    # o3 = i3

    o1 = pooling(o1)  # 全局平均池化层
    # o1 = BatchNormalization(o1)
    o1 = output(o1)  # 有1024个节点的全连接层
    # o1 = NormLayer(o1)
    #o1 = Dropout(0.1)(o1)

    o2 = pooling(o2)  # 全局平均池化层
    # o2 = BatchNormalization(o2)
    o2 = output(o2)  # 有1024个节点的全连接层
    #o2 = Dropout(0.1)(o2)
    # o2 = NormLayer(o2)

    o3 = pooling(o3)  # 全局平均池化层
    # o3 = BatchNormalization(o3)
    o3 = output(o3)  # 有1024个节点的全连接层
    # o3 = NormLayer(o3)

    #o3 = Dropout(0.1)(o3)

    #print('[INFO] base_model_layers', len(encoder.layers))
    def l2_normalize(x):
        return K.expand_dims(K.l2_normalize(x, 1))
    def l2_norm(x):
        return K.sqrt(K.sum(K.square(x), 1))

    def distance(inputs):
        ap, an, margin, gthr = inputs
        ap_l2n = K.sqrt(K.sum(K.square(ap), axis=1, keepdims=True))
        an_l2n = K.sqrt(K.sum(K.square(an), axis=1, keepdims=True))
        d = K.minimum((an_l2n - ap_l2n), margin)
        # d=an_l2n
        # g=ap_l2n
        g = K.maximum(ap_l2n, gthr)
        y = K.concatenate([d, g], axis=1)
        return y

    ap = Subtract()([o1, o2])
    an = Subtract()([o1, o3])

    val = Lambda(distance, name='margin')([ap, an, t1, t2])
    # val = Concatenate()([d, g])
    model = Model(inputs=[i10,i20,i30,t1,t2], outputs=val)
    # K.clear_session()
    return model


'''if os.path.exists(restore):
    model = models.load_model(restore)
    # 若成功加载前面保存的参数，输出下列信息
    print("[INFO] Model loaded from checkpoint!")
else:
    model = define_model()
    print("[INFO] Model loaded from pre-trained InceptionV3!")'''

# model = define_model()
'''model2 = models.load_model(restore, custom_objects={'max_error': max_error})
model.summary()
#model.set_weights(model2.get_weights())
encoder = model.get_layer('inception_v3')
encoder2 = model2.get_layer('inception_v3')
output = model.get_layer('encoder_output')
output2 = model2.get_layer('encoder_output')

encoder.set_weights(encoder2.get_weights())
output.set_weights(output2.get_weights())'''

#model.load_weights(restore, by_name=True)

#model.summary()
#model2.summary()
#parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model = model
#
#
# checkpoint = BaseModelCheckpoint(restore, base_model=model, monitor='val_loss', save_weights_only=False,verbose=1,save_best_only=False, period=1)
# tensorboard_r = TensorBoard(log_dir='log')
#
# #parallel_model = multi_gpu_model(model, gpus=2)
#
# opt = SGD(lr=0, momentum=0.9)
#opt = Adam(lr=1e-3, decay=1e-5)

# 分别使用transger learning和fine tune两种方法进行配置
# 1.设置所有层都不可训练
def setup_to_transfer_learning(model, models, opt):
    for base_model in models:
        print(len(base_model.layers))
        for layer in base_model.layers:
            layer.trainable = False
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.compile(optimizer=opt, loss='mse',  metrics=['mae', max_error])

# 2.设置基础模型的前面基层不可训练，后面的Inception模块可训练
def setup_to_fine_tune(model, models, opt):
    # GAP_LAYER = 250  # 需要根据实际效果进行确定
    # for base_model in models:
    #     for layer in base_model.layers[:GAP_LAYER + 1]:
    #         layer.trainable = False
    #     for layer in base_model.layers[GAP_LAYER + 1:]:
    #         layer.trainable = True
    # model.compile(optimizer=opt, loss='mse', metrics=['mae', max_error])
    for base_model in models:
        trainable = False
        for i, layer in enumerate(base_model.layers):
            layer.trainable = trainable
            if layer.name == 'mixed8':
                trainable = True

    model.compile(optimizer=opt, loss='mse', metrics=['mae', max_error])

#encoder = parallel_model.get_layer('model_1').get_layer('inception_v3')


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, variable_names_whitelist=freeze_var_names)
        return frozen_graph

def convert_frozen_pb_to_savedmodel(model, frozen_model_path, frozen_model_name, savedmodel_path):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
    from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

    '''graph_pb = frozen_model_path + frozen_model_name
    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_path)
    
    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())'''
    builder = tf.saved_model.builder.SavedModelBuilder(savedmodel_path)
    signature = {}

    #with tf.Session(graph=tf.Graph()) as sess:
    with K.get_session() as sess:
        # name="" is important to ensure we don't get spurious prefixing
        #tf.import_graph_def(graph_def, name="")
        K.set_learning_phase(0)
        encoder = model.get_layer('inception_v3')
        encoder_out = model.get_layer('encoder_output')

        graph_def = freeze_session(sess, output_names=[model.output.op.name])
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()

        #input1 = g.get_tensor_by_name(encoder.get_input_at(0).name)
        input1 = g.get_tensor_by_name(model.input[0].name)
        print('[INFO] Input tensor name:', input1.name)
        print('[INFO] Input tensor shape:', input1.shape)
        output1 = g.get_tensor_by_name(encoder_out.get_output_at(0).name)
        print('[INFO] Output tensor name:', output1.name)
        print('[INFO] Output tensor shape:', output1.shape)

        signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.saved_model.signature_def_utils.predict_signature_def(
                                                    {PREDICT_INPUTS: input1},
                                                    {PREDICT_OUTPUTS: output1})

        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=signature
                                            )
        builder.save()


'''root_path = './'
weight_file = './triplet_model_latest.h5'
frozen_model_name = 'tensor_model_v3.pb'
frozen_model_path = root_path + 'frozen/'
savedmodel_path = root_path + 'savedmodel/1/'
model.summary()

#input = model.input[0]
encoder = model.get_layer("inception_v3")    
encoder_out = model.get_layer("global_average_pooling2d_1")
input = encoder.get_input_at(0)
output = encoder.get_output_at(0)
model2 = Model(inputs=input, outputs=output)
#model2.compile(optimizer='sgd', loss='mse')
x1 = np.random.random((1, 300, 300, 3))
print(x1)
y = model2.predict(x1)
print(y)

#print(model.get_config())
#export_frozen_pb(model, frozen_model_path, frozen_model_name)
#convert_frozen_pb_to_savedmodel(model, frozen_model_path, frozen_model_name, savedmodel_path)'''


def sample_people(dataset):
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while i < nrof_classes:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        pic_name=dataset[class_index].getname()
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = nrof_images_in_class
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [pic_name] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1


    return image_paths, num_per_class,sampled_class_indices


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

    def getname(self):
        return self.name

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for cl in classes:
        path_peo=os.path.join(path_exp,cl)
        people=[path for path in os.listdir(path_peo) if os.path.isdir(os.path.join(path_peo, path))]
        people.sort()
        nrof_peoples=len(people)
        for i in range(nrof_peoples):
            class_name = people[i]
            facedir = os.path.join(path_peo, class_name)
            name=cl+'_'+class_name
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def parse_arguments(argv):

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="./paths.txt",
                    help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-a", "--annotation", default="./txt",
                    help="path to input dataset (i.e., directory of annotation file)")
    ap.add_argument("-m", "--model", default="face.model",
                    help="path to output model")
    ap.add_argument("-l", "--labelbin", default="face.pickle",
                    help="path to output label binarizer")
    ap.add_argument("-p", "--plot", type=str, default="plot2.png",
                    help="path to output accuracy/loss plot")
    ap.add_argument("-v", "--vec_len", type=int, default=256,
                    help="length of feature vector")
    ap.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default=r'E:\train_TEST')
    ap.add_argument('--weight_dir', type=str,
                    help='Weights restored from pre-trained InceptionV3.',
                    default=r'F:\KerasTriplet\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    ap.add_argument('--MARGIN', type=float,
                    help='distince of (an-ap)',
                    default=10.0)
    ap.add_argument('--GAP', type=float,
                    help='distince of ap',
                    default=1.0)
    ap.add_argument('--mf', type=str,
                    help='model file',
                    default=r'E:\Keras_triplet_model_0903.h5')
    ap.add_argument('--test', type=str,
                    help='test file',
                    default=r'F:\train_Test_fill')

    # args = vars(ap.parse_args())
    #
    return ap.parse_args(argv)


def main(args):
    VEC_LEN = args.vec_len
    NEG_TIMES_OF_POS = 1
    EPOCHS = 100
    WORKERS = 20
    INIT_LR = 1e-3
    BS = 30
    IMAGE_DIMS = (178, 218, 3)
    # MARGIN = math.sqrt(VEC_LEN * 4.) * 0.8
    # GAP = MARGIN * 0.5
    MARGIN=args.MARGIN
    GAP=args.GAP
    restore="triplet_Adagrad_latest"+'_'+str(int(MARGIN))+'_'+str(int(GAP))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    print(args)
    tf.Session(config=config)
    dataset = get_dataset(args.data_dir)
    image_paths, num_per_class, sampled_class_indices = sample_people(dataset)
    # generate_arrays(image_paths, BS,MARGIN,GAP,IMAGE_DIMS,sampled_class_indices)

    model = define_model(IMAGE_DIMS,VEC_LEN,args.weight_dir)
    # data = generate_arrays(image_paths, BS, MARGIN, GAP, IMAGE_DIMS, sampled_class_indices)
    # model = load_model('triplet_model_latest.h5', custom_objects={'max_error': max_error})
    # model.summary()
    # for sample in data:
    #     x, y = sample
        # X1, X2, X3, Y1, Y2=x
        # print(x)
        # y_pre=model.predict(x)
        # print(y_pre,y)

    parallel_model = model

    checkpoint = BaseModelCheckpoint(restore, base_model=parallel_model, monitor='val_loss', save_weights_only=False, verbose=1,
                                     save_best_only=False, period=1)
    tensorboard_r = TensorBoard(log_dir='log')
    # opt = SGD(lr=0.01, momentum=0.1)
    opt=Adam(lr=0.001,epsilon=1e-08)
    # opt=Adagrad(lr=0.01)
    encoder = model.get_layer('inception_v3')

    # setup_to_transfer_learning(parallel_model, [encoder], opt)
    setup_to_fine_tune(parallel_model, [encoder],opt)

    # parallel_model.summary()

    # train the network
    print("[INFO] training network...")

    random.seed(int(time.time()))

    H = parallel_model.fit_generator(
        # H = model.fit_generator(
        generate_arrays(image_paths, BS,MARGIN,GAP,IMAGE_DIMS,sampled_class_indices),
        # validation_data=(testX, testY),
        steps_per_epoch=len(image_paths) // BS,
        # steps_per_epoch=100,
        workers=WORKERS, use_multiprocessing=True,
        epochs=EPOCHS, verbose=1, callbacks=[checkpoint, tensorboard_r])
        # epochs=EPOCHS, verbose=1)

    print("[INFO] Training done")



def acc(args):
    VEC_LEN = 256
    BS = 5
    IMAGE_DIMS = (300, 300, 3)
    MARGIN = 10.0
    GAP = 1.0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    tf.Session(config=config)
    test=args.test
    dataset = get_dataset(test)
    image_paths, num_per_class, sampled_class_indices = sample_people(dataset)
    # data=generate_arrays(image_paths, BS, MARGIN, GAP, IMAGE_DIMS, sampled_class_indices)
    # model = define_model(IMAGE_DIM
    # S, VEC_LEN, args.weight_dir)
    imgsize=len(image_paths)
    print(imgsize)
    model_path=args.mf
    # resultpath='/home/hzp/Downloads/savemodel'
    resultname=re.split('/|\.',model_path)[-2]+'.txt'
    # resultofmodel=os.path.join(resultpath,resultname)

    # print(resultname)
    model = load_model(model_path, custom_objects={'max_error': max_error})
    # model = model.load_weights('/home/hzp/Downloads/savemodel/triplet_model_latest.h5')
    # for item in model.layers:
    #     print(item.name)
    model.summary()
    glo_av=Model(inputs=model.input[0],outputs=model.get_layer('encoder_output').output)
    all_pre=[]
    count=0
    X=[]
    sampe_peo=[]
    for i in range(imgsize):
        # print(image_paths[i])
        pic=read_pic(image_paths[i], IMAGE_DIMS)
        if pic is None:
            continue
        sampe_peo.append(sampled_class_indices[i])
        X.append(pic)
        if len(X)==5 or i==imgsize-1:
            pic = (np.array(X, dtype="float32") - 127.5) / 127.5
            y_pre=glo_av.predict(pic)
            all_pre.extend(y_pre.tolist())
            X=[]
        else:
            continue

        count+=1
        print(count)
        # if count==5:
        #     break
    sample_size=len(sampe_peo)
    print(sample_size)
    y_label=np.zeros((sample_size,sample_size))
    for i in range(sample_size):
        for j in range(i+1,sample_size):
            if sampe_peo[i]==sampe_peo[j]:
                y_label[i][j]=1
    print(y_label.shape)
    # A是一个向量矩阵：euclidean代表欧式距离
    distA = pdist(all_pre, metric='euclidean')
    # 将distA数组变成一个矩阵
    distB = squareform(distA)
    # all=pd.DataFrame(all_pre)
    print(distB.shape)
    step=np.max(distB)/10.0
    # all.to_excel('ex0826.xls')
    '''         pre
                   1     0
          act  1  tp    fn  召回率
               0  fp    tn
                精确率
        '''
    assert y_label.shape[0]==distB.shape[0]
    for threshold in range(1,10):
        threshold=threshold*step
        tp,fp,fn,tn=0,0,0,0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                if y_label[i][j]==1:
                    if distB[i][j] <= threshold:
                        tp+=1
                    else:
                        fn+=1
                if y_label[i][j]==0:
                    if distB[i][j] <= threshold:
                        fp+=1
                    else:
                        tn+=1
        tpr = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) != 0.0 else 0  # 正——召回率
        tpa = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) != 0.0 else 0  # 正——准确率
        tfr = float(tn) / (float(tn) + float(fp)) if (float(tn) + float(fp)) != 0.0 else 0  # 负——召回率
        tfa = float(tn) / (float(tn) + float(fn)) if (float(tn) + float(fn)) != 0.0 else 0  # 负——准确率
        print(tp,fp,fn,tn)
        print(tpr,tpa,tfr,tfa)
        with open(resultname, "a") as f:
            f.write("test size : {}\n".format(sample_size))
            f.write("thresold value : {}\n".format(threshold))
            f.write("tp : {}      fn : {}\n".format(tp,fn))
            f.write("fp : {}      tn : {}\n".format(fp,tn))
            f.write("正样本召回率：%.9f 正样本准确率：%.9f 负样本召回率：%.9f 负样本准确率：%.9f\n"%(tpr,tpa,tfr,tfa))



if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))
    # print(parse_arguments(sys.argv[1:]))
    # acc(parse_arguments(sys.argv[1:]))
