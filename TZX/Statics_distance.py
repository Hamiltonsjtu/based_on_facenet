"""
This function tries to find the statistics information of the embedding vector for specific person.
Which is helpful to set a suitable margin of recognition
Author: Shuai Zhu @TZX
Time: 2019/05/22 16:36
"""

import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as st

# sys.path.append("../facenet/src") # useful for the import of facenet in another folder
# import align.detect_face

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
DATA = np.load('people_embs.npy').item()


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

emb_data = np.load('people_embs.npy').item()

# print(cal_likely(1.554))
# print(cal_likely(-1.554))
#
people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
# attrib = ['emb', 'distance', 'average_emb']
# emb_data = multi(people, attrib, {})
for i in people:
    # for j in attrib:
    #     emb_data[i][j] = np.loadtxt('data/images_cropped/' + i + '/' + j + '.txt', delimiter=' ')
    # print('num of standard images for {} is {}'.format(i, np.shape(emb_data[i]['emb'])))
    emb_data[i]['dist_emb'] = [np.sqrt(np.sum(np.square(np.subtract(k, emb_data[i]['average_emb'])))) for k in emb_data[i]['emb']]
    emb_data[i]['log_dist'] = np.log(emb_data[i]['dist_emb'])
    emb_data[i]['log_dist_mean'] = np.mean(emb_data[i]['log_dist'])
    emb_data[i]['log_dist_std'] = np.std(emb_data[i]['log_dist'])
    emb_data[i]['boxcox_dist_emb_aver'] = st.boxcox(np.abs(emb_data[i]['dist_emb']))
    # print('distance shape is {}'.format(np.shape(emb_data[i]['distance'])))


    # nn = np.shape(emb_data[i]['distance'])[0]
    # emb_data[i]['log_all_dist'] = [np.log(emb_data[i]['distance'][kk,:]) for kk in range(nn)]
    # emb_data[i]['log_all_dist'][np.isnan(emb_data[i]['log_all_dist'])] = 0
    # db = DBSCAN(eps=0.7, min_samples=1, metric='precomputed')
    # db.fit(emb_data[i]['log_all_dist'])
    # labels = db.labels_
    # no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print('DBSACN: {} has {} of clusters:'.format(i, no_clusters))

#
# emb_ = np.array([np.loadtxt('hu_emb.txt', delimiter=' ')])
# # emb_ = []
#
# for i in people:
#     emb_add_face = np.vstack((emb_data[i]['emb'], emb_))
#     n, m = np.shape(emb_add_face)
#     print('emb_add_face shape {}'.format(np.shape(emb_add_face)))
#     dist_add_face = np.zeros((n, n))
#     for ii in range(n):
#         for jj in range(n):
#             dist_add_face[ii,jj] = np.sqrt(np.sum(np.square(np.subtract(emb_add_face[ii,:], emb_add_face[jj,:]))))
# # estimator = GaussianMixture(n_components=7, covariance_type='full')
# # estimator.fit(dist_add_face)
# # pred = estimator.predict(dist_add_face)
# # # print('================ {} ===================='.format(i))
# # print(pred)
#     db = DBSCAN(eps=0.75, min_samples=1, metric='precomputed')
#     db.fit(dist_add_face)
#     labels = db.labels_
#     no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print('DBSACN: {} has {} of clusters:'.format(i, no_clusters))

# emb_ = np.loadtxt('xi2017_emb.txt', delimiter=' ')
# n = np.shape(emb_)[0]
# if n == 128:
#     print('======== Only one face in picture DISTANCE ========')
#     emb_jack = emb_
#     jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
#     for i in people:
#         jack_dist[i]['dist_average'] = [np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
#         jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
#         jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean'])/emb_data[i]['log_dist_std']
#         jack_dist[i]['Prob'] = 2*st.norm.cdf(-np.abs(jack_dist[i]['Z_value']))
#         print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
#         print('The face is {} likely {}'.format(jack_dist[i]['Prob'], i))
# else:
#     for l in range(n):
#         print('======== START CALCULATE THE {}-th face DISTANCE ========'.format(l))
#         emb_jack = emb_[l,:]
#         jack_dist = multi(people, ['dist_all', 'dist_average', 'dist_all_average'], {})
#         for i in people:
#             jack_dist[i]['dist_average'] = [
#                 np.sqrt(np.sum(np.square(np.subtract(emb_jack, emb_data[i]['average_emb']))))]
#             jack_dist[i]['log_dist_average'] = np.log(jack_dist[i]['dist_average'])
#             jack_dist[i]['Z_value'] = (jack_dist[i]['log_dist_average'] - emb_data[i]['log_dist_mean'])/emb_data[i]['log_dist_std']
#             jack_dist[i]['Prob'] = 2*st.norm.cdf(-np.abs(jack_dist[i]['Z_value']))
#             print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
#             print('The face is {} likely {}'.format(jack_dist[i]['Prob'], i))
            # print('Z value for {} is {}'.format(i, jack_dist[i]['Z_value']))
            # print('The face is {} likely {}'.format(jack_dist[i]['Prob'], i))

#
#     # print('dist_average for {} is {}'.format(i, jack_dist[i]['dist_average']))
#     # print('=============== DBSCAN ===============')
#     # # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
#     # matrix = emb_data[i]['distance']
#     # db = DBSCAN(eps=0.75, min_samples=1, metric='precomputed')
#     # db.fit(matrix)
#     # labels = db.labels_
#     # no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     # print('DBSACN: {} has {} of clusters:'.format(i, no_clusters))

# attrib = ['emb', 'distance', 'average_emb']
# people = ['ALL']
# emb_data_test = multi(people, attrib, {})

# for i in people:
#     for j in attrib:
#         emb_data_test[i][j] = np.loadtxt('data/images_cropped/' + i + '/' + j + '.txt', delimiter=' ')
#     print('=============== DBSCAN ===============')
#     # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
#     matrix = emb_data_test['ALL']['distance']
#     db = DBSCAN(eps=0.8, min_samples=1, metric='precomputed')
#     db.fit(matrix)
#     labels = db.labels_
#     print('============================')
#     print(labels)
#     no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print('ALL has {} of clusters:'.format(no_clusters))
#
#     print('=============== GMM ===============')
#     estimator = GaussianMixture(n_components=7, covariance_type='full')
#     dist_matrix = emb_data_test['ALL']['distance']
#     estimator.fit(dist_matrix)
#     pred = estimator.predict(dist_matrix)
#     proba = estimator.predict_proba(dist_matrix[10:])
#     scores = estimator.score_samples(dist_matrix)
#     print('pred', pred)
#     # print('proba', proba)
#     # print('scores', scores)
#     # print('n_components', estimator.n_components)
#     # print('means', estimator.means_)
#     e_mean = estimator.means_
#     no_clusters = len(set(pred)) - (1 if -1 in pred else 0)
#     print('number of cluster ', no_clusters)

#
# print('==============++++++++++++===============')
# for i in people:
#     print('the distance for {} is {}'.format(i, np.max(emb_data[i]['dist_emb'])))

    # print(emb_data[i]['dist_emb'])
# print('=============xijinping emb arr==============')
# print(emb_data['xijinping']['emb'])
# print('=============xijinping average emb ==============')
# print(emb_data['xijinping']['average_emb'])
# print('====================')
# print('emb_data key {}, emb_vector_shape {} and average emb_vector shape {}'.format(emb_data['xijinping'].keys(), np.shape(emb_data['xijinping']['emb']),np.shape(emb_data['xijinping']['average_emb'])))

# !!!!!!!!!!! CALCULATE AVERAGE EMBEDDING VECTOR BEFORE DISTANCE DOES NOT MAKE SENSE !!!!!!!!!
# dist_diff = {}
# for i in people:
#     dist_diff[i] = calculate_distance_emb_diff(emb_data[i]['emb'], emb_data[i]['average_emb'])
#     print('location \'' + i + '\' is {}'.format(np.where(dist_diff[i] >= np.max(dist_diff[i]))[0]))
# print(dist_diff['xijinping'])



for i in people:
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    plt.hist(emb_data[i]['log_dist'], histtype='bar', align='mid', orientation='vertical')
    ax1.set_title('log distribution')

    ax2 = fig.add_subplot(222)
    plt.hist(emb_data[i]['boxcox_dist_emb_aver'], histtype='bar', align='mid', orientation='vertical')
    ax2.set_title('boxcox distribution')

    ax3 = fig.add_subplot(223)
    st.probplot(emb_data[i]['log_dist'])
    ax3.set_title('log norm check')

    # ax4 = fig.add_subplot(224)
    # st.probplot(emb_data[i]['boxcox_dist_emb_aver'])
    # ax4.set_title('boxcox norm check')


    plt.show()
# plt.hist((dist_matrix - np.mean(dist_matrix))/np.std(dist_matrix), histtype='bar', align='mid', orientation='vertical')

# ax1 = plt.subplot(2,4,2)
# dist_matrix = emb_data['xijinping']['dist_emb']
# plt.hist(dist_matrix, histtype='bar', align='mid', orientation='vertical')

# ax2 = plt.subplot(2,4,2)
# plt.hist(emb_data['hujintao'], histtype='bar', align='mid', orientation='vertical')
# ax3 = plt.subplot(2,4,3)
# plt.hist(dist_diff['jiangzemin'], histtype='bar', align='mid', orientation='vertical' )
# ax3.set_title('jiangzemin'+str(np.where(dist_diff['jiangzemin'] >= np.max(dist_diff['jiangzemin'] ))[0]))
# ax4 = plt.subplot(2,4,4)
# plt.hist(dist_diff['dengxiaoping'], histtype='bar', align='mid', orientation='vertical' )
# ax4.set_title('dengxiaoping'+str(np.where(dist_diff['dengxiaoping'] >= np.max(dist_diff['dengxiaoping'] ))[0]))
# ax5 = plt.subplot(2,4,5)
# plt.hist(dist_diff['wenjiabao'], histtype='bar', align='mid', orientation='vertical' )
# ax5.set_title('wenjiabao' + str(np.where(dist_diff['wenjiabao'] >= np.max(dist_diff['wenjiabao'] ))[0]))
# ax6 = plt.subplot(2,4,6)
# plt.hist(dist_diff['maozedong'], histtype='bar', align='mid', orientation='vertical' )
# ax6.set_title('maozedong' + str(np.where(dist_diff['maozedong'] >= np.max(dist_diff['maozedong'] ))[0]))
# ax7 = plt.subplot(2,4,7)
# plt.hist(dist_diff['zhouenlai'], histtype='bar', align='mid', orientation='vertical' )
# ax7.set_title('zhouenlai' + str(np.where(dist_diff['zhouenlai'] >= np.max(dist_diff['zhouenlai'] ))[0]))
# plt.show()
