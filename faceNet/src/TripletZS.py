from tensorflow.python.keras.optimizers import SGD, Adam, Adagrad
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
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, AveragePooling2D, Flatten, Softmax, Subtract, \
    Lambda, Reshape, Minimum, Concatenate
from tensorflow.python.keras.models import Model, load_model
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.ops.linalg.linalg_impl import norm
import imageio
from sklearn.model_selection import KFold

class Sequ_pic(Sequence):
    def __init__(self, triplets, MARGIN, GAP, IMAGE_DIMS, batch_size):
        self.triplets = triplets
        self.MARGIN = MARGIN
        self.GAP = GAP
        self.IMAGE_DIMS = IMAGE_DIMS
        self.batch_size = batch_size
    def __len__(self):
        # print('-----------------')
        # print(len(self.triplets))
        return int(np.ceil(len(self.triplets) / float(self.batch_size)))
    def __getitem__(self, idx):
        train_x = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = self.data_generation(train_x)
        return x,y
    def data_generation(self, train_x):
        X1 = []
        X2 = []
        X3 = []
        Y1 = []
        Y2 = []
        Y = []
        for i in range(len(train_x)):
            x1 = self._read_pic(train_x[i][0])
            x2 = self._read_pic(train_x[i][1])
            x3 = self._read_pic(train_x[i][2])
            if x1 is None:
                continue
            if x2 is None:
                continue
            if x3 is None:
                continue
            X1.append(x1)
            X2.append(x2)
            X3.append(x3)
            Y1.append([self.MARGIN])
            Y2.append([self.GAP])

            Y.append([self.MARGIN, self.GAP])
        return [np.array(X1), np.array(X2), np.array(X3), np.array(Y1), np.array(Y2)], np.array(Y)

    def _read_pic(self, path):
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - 127.5) / 127.5
        except Exception as e:
            img = None
        return img


def read_pic(path):
        img_tmp = cv2.imread(path)
        img_tmp = img_tmp[:,:,::-1]
        img = (img_tmp - 127.5) / 127.5
        return img


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
        pic_name = dataset[class_index].getname()
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = nrof_images_in_class
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [pic_name] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class, sampled_class_indices


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
    # print('path is {}'.format(path))
    path_exp = os.path.expanduser(path)
    # print('path expanduser is {}'.format(path_exp))
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    print('have {} class(es), and class is {}'.format(nrof_classes, classes))
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def select_triplets(embeddings, sam_lab, image_paths, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    while emb_start_idx < len(sam_lab):
        same_people = [j for j, v in enumerate(sam_lab) if v == sam_lab[emb_start_idx]]
        if len(same_people) < 2:
            emb_start_idx += len(same_people)
            continue
        for j in range(1, len(same_people)):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sqrt(np.sum(np.square(embeddings[a_idx] - embeddings), 1))
            for pair in range(j, len(same_people)):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sqrt(np.sum(np.square(embeddings[a_idx] - embeddings[p_idx])))
                neg_dists_sqr[emb_start_idx:emb_start_idx + len(same_people)] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d)' %
                       (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += len(same_people)

    np.random.shuffle(triplets)
    return triplets


def max_error(y_true, y_pred):
    '''diff = K.abs(y_true - y_pred)
    K.max(Lambda(x:))'''
    return K.max(K.abs(y_true - y_pred))


def abs_pre(y_true, y_pred):
    return K.abs(y_pred)


def define_model(IMAGE_DIMS, VEC_LEN, weight_dir):
    i10 = Input(shape=IMAGE_DIMS)
    i20 = Input(shape=IMAGE_DIMS)
    i30 = Input(shape=IMAGE_DIMS)
    t1 = Input(shape=(1,))
    t2 = Input(shape=(1,))

    print("[INFO] Weights restored from pre-trained InceptionV3!")
    encoder = InceptionV3(weights=weight_dir, include_top=False)
    pooling = GlobalAveragePooling2D()

    def l2_norm_(x):
        return K.sqrt(K.sum(K.square(x), 1))

    def l2_normalize(x):
        return K.l2_normalize(x, 1)

    output = Dense(VEC_LEN, activation='sigmoid', name='encoder_output')

    o1 = encoder(i10)
    o2 = encoder(i20)
    o3 = encoder(i30)

    o1 = pooling(o1)  # 全局平均池化层
    o1 = output(o1)  # 有1024个节点的全连接层

    o2 = pooling(o2)  # 全局平均池化层
    o2 = output(o2)  # 有1024个节点的全连接层

    o3 = pooling(o3)  # 全局平均池化层
    o3 = output(o3)  # 有1024个节点的全连接层

    def distance(inputs):
        ap, an, margin, gthr = inputs
        ap_l2n = K.sqrt(K.sum(K.square(ap), axis=1, keepdims=True))
        an_l2n = K.sqrt(K.sum(K.square(an), axis=1, keepdims=True))
        d = K.minimum((an_l2n - ap_l2n), margin)
        g = K.maximum(ap_l2n, gthr)
        y = K.concatenate([d, g], axis=1)
        return y

    ap = Subtract()([o1, o2])
    an = Subtract()([o1, o3])

    val = Lambda(distance, name='margin')([ap, an, t1, t2])
    model = Model(inputs=[i10, i20, i30, t1, t2], outputs=val)

    return model


# 1.设置所有层都不可训练
def setup_to_transfer_learning(model, models, opt):
    for base_model in models:
        print(len(base_model.layers))
        for layer in base_model.layers:
            layer.trainable = False
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.compile(optimizer=opt, loss="mse", metrics=['mae', max_error])


# 2.设置基础模型的前面基层不可训练，后面的Inception模块可训练
def setup_to_fine_tune(model, models, opt):
    for base_model in models:
        trainable = False
        for i, layer in enumerate(base_model.layers):
            layer.trainable = trainable
            if layer.name == 'mixed8':
                trainable = True
    model.compile(optimizer=opt, loss="mse", metrics=['mae', max_error])


def main(args):
    VEC_LEN = args.vec_len
    NEG_TIMES_OF_POS = 1
    EPOCHS = 60
    WORKERS = 10
    INIT_LR = 1e-3
    batch_size = 2000
    IMAGE_DIMS = (300, 300, 3)
    GAP = args.GAP
    MARGIN = args.Margin
    restore = "triplet_SGDhard_latest" + '_' + str(int(MARGIN)) + '_' + str(int(GAP))
    config = tf.ConfigProto()
    tf.Session(config=config)
    dataset = get_dataset(args.data_dir)
    image_paths, num_per_class, sampled_class_indices = sample_people(dataset)

    model = define_model(IMAGE_DIMS, VEC_LEN, args.weight_dir)
    parallel_model = model
    opt = SGD(lr=0.01, momentum=0.1)
    encoder = model.get_layer('inception_v3')
    setup_to_fine_tune(model, [encoder], opt)
    model.summary()


    print("[INFO] training network...")
    steps_per_epoch = int(np.ceil(len(image_paths) / batch_size))
    step = 0
    while step < EPOCHS:
        triplets = []
        for i in range(steps_per_epoch):
            print(i)
            if i == steps_per_epoch - 1:
                sample_image = image_paths[batch_size * i:]
                sample_class = sampled_class_indices[batch_size * i:]
            else:
                sample_image = image_paths[batch_size * i:batch_size * (i + 1)]
                sample_class = sampled_class_indices[batch_size * i:batch_size * (i + 1)]
            pic_list = []
            sam_lab = []
            samimg_path = []
            for j in range(len(sample_image)):
                x = read_pic(sample_image[j])
                if x is None:
                    continue
                else:
                    pic_list.append(x)
                    sam_lab.append(sample_class[j])
                    samimg_path.append(image_paths[j])

            pic_list = np.array(pic_list, dtype="uint8")
            glo_av = Model(inputs=parallel_model.input[0], outputs=parallel_model.get_layer('encoder_output').output)
            embs = glo_av.predict(pic_list)
            triplets.extend(select_triplets(embs, sam_lab, samimg_path, GAP))

        generate_data = Sequ_pic(triplets, MARGIN, GAP, IMAGE_DIMS, batch_size)
        parallel_model.fit_generator(generate_data, steps_per_epoch=int(np.ceil(len(triplets)/60)), workers=WORKERS, use_multiprocessing=True)

        if not os.path.exists(restore):
            os.makedirs(restore)
        fname = restore + '_' + str(step + 1) + 'epoch.h5'
        fp = os.path.join(restore, fname)
        print('\nEpoch %05d: saving model to %s' % (step + 1, fp))
        model.save(fp, overwrite=True)
        step += 1
    print("[INFO] Training done")


def acc(args):
    print('-----  Start Loading Model  --------')
    starttime = time.time()
    trainedModel = load_model(args.trained_model, custom_objects={'max_error': max_error})
    print('-----  Load Model Cost {}s --------'.format(time.time()-starttime))
    trainedModel.summary()
    glo_av = Model(inputs= trainedModel.input[0], outputs=trainedModel.get_layer('encoder_output').output)

    ########################################################################
    image_paths, actual_issame = get_test_pairs(args.val_dir, 100)
    nrof_images = len(image_paths)
    img_list = []
    for i in image_paths:
        img_tmp = cv2.resize(imageio.imread(i), (160, 160), interpolation=cv2.INTER_AREA)
        img_list.append(img_tmp)
    emb_array = np.zeros((nrof_images, args.vec_len))
    ########################################################################
    # Run forward pass to calculate embeddings
    if nrof_images % args.bs == 0:
        nrof_batches = nrof_images // args.bs
    else:
        nrof_batches = (nrof_images // args.bs) + 1
    print('Number of batches: ', nrof_batches)
    for i in range(nrof_batches):
        if i == nrof_batches - 1:
            n = nrof_images
        else:
            n = i * args.bs + args.bs
        img_emb = img_list[i * args.bs:n]
        embs = glo_av.predict(img_emb)
        emb_array[i * args.bs:n, :] = embs

    tp, fp, tn, fn,  accuracy = evaluate(emb_array, actual_issame, nrof_folds=args.lfw_nrof_folds)
    # _, _, accuracy, val, val_std, far = evaluate(embeddings, actual_issame, nrof_folds=args.lfw_nrof_folds)
    anchor = emb_array[0:len(actual_issame):2]
    postiv = emb_array[1:len(actual_issame):2]
    negtiv = emb_array[len(actual_issame):len(image_paths):2]
    pos_dist = (np.sum(np.square(anchor - postiv), 1))/len(anchor)
    neg_dist = (np.sum(np.square(anchor - negtiv), 1))/len(anchor)
    basic_loss = pos_dist - neg_dist + 0.2
    loss = (np.sum(np.max(basic_loss, 0)))/len(anchor)
    print('tp:{} and \t fp:{}'.format(tp, fp))
    print('fn:{} and \t tn:{}'.format(fn, tn))
    print('accuracy: {}'.format(accuracy))
    print('mean accuracy: {}'.format(np.mean(accuracy)))
    print('TPR: ', np.sum(tp)/(np.sum(tp) +np.sum(fn)))
    print('FPR: ', np.sum(fp)/(np.sum(fp) +np.sum(tn)))
    print('FNR: ', np.sum(fn)/(np.sum(tp) +np.sum(fn)))
    print('TNR: ', np.sum(tn)/(np.sum(fp) +np.sum(tn)))
    print('Loss: ', loss)



def get_test_pairs(tzx_dir, nnum):
    '''
    :param tzx_dir:  dir of image
    :param nnum:  number of pairs
    :return: image list with size 2*nnum and label list with size nnum
    '''
    path_list = []
    issame_list = []

    sl_peo_t = np.random.randint(0, len(os.listdir(tzx_dir)), nnum)
    for i in sl_peo_t:
        peo_dir = os.listdir(tzx_dir)
        peo_path = os.path.join(tzx_dir, peo_dir[i])
        img_dir = os.listdir(peo_path)
        # print('peo_path num {} and its path {}'.format(len(os.listdir(peo_path)), peo_path))
        sl_img = random.sample(range(len(os.listdir(peo_path))), 2)
        img_path_0 = os.path.join(peo_path, img_dir[sl_img[0]])
        img_path_1 = os.path.join(peo_path, img_dir[sl_img[1]])
        path_list.append(img_path_0)
        path_list.append(img_path_1)
        issame_list.append(True)

    for i in range(nnum):
        sl_peo_f = random.sample(os.listdir(tzx_dir), 2)
        for j in sl_peo_f:
            peo_dir = os.path.join(tzx_dir, j)
            img_sl = random.sample(os.listdir(peo_dir), 1)
            img_dir = os.path.join(peo_dir, img_sl[0])
            path_list.append(img_dir)
        issame_list.append(False)

    return path_list, issame_list


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sqrt(np.sum(np.square(diff), 1))
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=1,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    tp = np.zeros((nrof_folds))
    tn = np.zeros((nrof_folds))
    fp = np.zeros((nrof_folds))
    fn = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print('fold_idx: {} and indeices: {}'.format(fold_idx, train_set))
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print('fold_idx: {} best acc: {} and its indices: {}'.format(fold_idx, acc_train[best_threshold_index], best_threshold_index))
        # for threshold_idx, threshold in enumerate(thresholds):
        #     tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
        #                                                                                          dist[test_set],
        #                                                                                          actual_issame[
        #                                                                                              test_set])
        tp[fold_idx], fp[fold_idx], tn[fold_idx], fn[fold_idx], accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[train_set],
                                                      actual_issame[train_set])

        # tpr = np.mean(tprs, 0)
        # fpr = np.mean(fprs, 0)
    return tp, fp, tn, fn, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tp, fp, tn, fn, acc

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(1, 5, 0.5)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tp, fp, tn, fn,  accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tp, fp, tn, fn, accuracy

def parse_arguments(argv):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vec_len", type=int, default=256,
                    help="length of feature vector")
    ap.add_argument('--data_dir', type=str,
                    help='Path to the data directory containing aligned face patches.',
                    # default=r'F:\train_Test')
                    default='/home/SAVE_PIC_fill')
    ap.add_argument('--weight_dir', type=str,
                    help='Weights restored from pre-trained InceptionV3.',
                    # default=r'F:\KerasTriplet\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
                    default='/home/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    ap.add_argument('--Margin', type=float,
                    help='distince of (an-ap)',
                    default=10)
    ap.add_argument('--GAP', type=float,
                    help='distince of ap',
                    default=3)

    ap.add_argument('--val_dir', type=str,
                    help='path to validation data',
                    default=r'F:\train_Test\610112003-1827-0274_8')
    ap.add_argument('--trained_model', type=str,
                    help='path to trained_model',
                    default=r'F:\triplet_SGDhard_latest_10_3_50epoch.h5')
    ap.add_argument("-b", "--bs", type=int, default=30)

    return ap.parse_args(argv)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    acc(parse_arguments(sys.argv[1:]))


