"""
This function tries to find the statistics information of the embedding vector for specific person.
Which is helpful to set a suitable margin of recognition
Author: Shuai Zhu @TZX
Time: 2019/06/20 16:36
"""

import scipy as sci
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import scipy.stats as st
import time
import argparse
sys.path.append("../facenet/src") # useful for the import of facenet in another folder
import align.detect_face
import facenet



def multi(*args):
    """
    Build multiple level dictionary for python
    For example:
        multi(['a', 'b'], ['A', 'B'], ['1', '2'], {})
    returns
        {   'a': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}},
            'b': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}}}
    """
    if len(args) > 1:
        return {arg:multi(*args[1:]) for arg in args[0]}
    else:
        return args[0]


def calculate_distance_emb_diff(emb, emb_average):
    """
    :param emb: the e
    :param emb_average:
    :return:
    """
    n = np.shape(emb)[0]
    print('have {} images'.format(n))
    dist = np.zeros((n, 1))
    for i in range(n):
        dist[i] = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb_average))))
    return dist


def cal_likely(Z_value):
    if Z_value == 0:
        likely = 1
    elif Z_value < 0:
        likely = 2*st.norm.cdf(Z_value)
    else:
        likely = 1 - st.norm.cdf(-np.abs(Z_value))
    return likely


# load critical embeddings
emb_data = np.load('people_embs.npy').item()
people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
for i in people:
    emb_data[i]['dist_emb'] = [np.sqrt(np.sum(np.square(np.subtract(k, emb_data[i]['average_emb'])))) for k in emb_data[i]['emb']]
    emb_data[i]['log_dist'] = np.log(emb_data[i]['dist_emb'])
    emb_data[i]['log_dist_mean'] = np.mean(emb_data[i]['log_dist'])
    emb_data[i]['log_dist_std'] = np.std(emb_data[i]['log_dist'])
    emb_data[i]['boxcox_dist_emb_aver'] = st.boxcox(np.abs(emb_data[i]['dist_emb']))


def cal_sim(emb, emb_data):
    people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
    def multi(*args):
        """
        Build multiple level dictionary for python
        For example:
            multi(['a', 'b'], ['A', 'B'], ['1', '2'], {})
        returns
            {   'a': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}},
                'b': {'A': {'1': {}, '2': {}}, 'B': {'1': {}, '2': {}}}}
        """
        if len(args) > 1:
            return {arg:multi(*args[1:]) for arg in args[0]}
        else:
            return args[0]

    emb_jack = emb
    jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
    likely = {}
    for i in people:
        jack_dist[i]['dist_average'] = [
            np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
        jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
        jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean']) / (
        emb_data[i]['log_dist_std'])
        jack_dist[i]['Prob'] = st.norm.pdf(jack_dist[i]['Z_value']) / st.norm.pdf(0)
        likely[i] = jack_dist[i]['Prob'][0]

    return likely


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model_dir)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # load image from path
            images = facenet.load_data(args.image_path, False, False, args.image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder:False }
            # Use the facenet model to calcualte embeddings
            embed = sess.run(embeddings, feed_dict=feed_dict)
            np.save('face_emb', embed)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_path', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))