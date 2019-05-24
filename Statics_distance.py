"""
This function tries to find the statistics information of the embedding vector for specific person.
Which is helpful to set a suitable margin of recognition
Author: Shuai Zhu @TZX
Time: 2019/05/22 16:36
"""

import scipy as sci
import numpy as np
import matplotlib.pyplot as plt


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


people = ['xijinping', 'hujintao', 'jiangzemin', 'dengxiaoping', 'wenjiabao', 'maozedong', 'zhouenlai']
attrib = ['emb', 'average_emb']
emb_data = multi(people, attrib, {})
for i in people:
    for j in attrib:
        emb_data[i][j] = np.loadtxt('data/images_cropped/'+i+'/'+j+'.txt', delimiter=' ')

# print('====================')
# print('emb_data key {}, emb_vector_shape {} and average emb_vector shape {}'.format(emb_data['xijinping'].keys(), np.shape(emb_data['xijinping']['emb']),np.shape(emb_data['xijinping']['average_emb'])))

dist_diff = {}
for i in people:
    dist_diff[i] = calculate_distance_emb_diff(emb_data[i]['emb'], emb_data[i]['average_emb'])
    print('location \'' + i + '\' is {}'.format(np.where(dist_diff[i] >= np.max(dist_diff[i]))[0]))
# print(dist_diff['xijinping'])
fig = plt.figure()
ax1 = plt.subplot(2,4,1)
plt.hist(dist_diff['xijinping'], histtype='bar', align='mid', orientation='vertical')
ax1.set_title('xijinping' + str(np.where(dist_diff['xijinping'] >= np.max(dist_diff['xijinping']))[0]))
ax2 = plt.subplot(2,4,2)
plt.hist(dist_diff['hujintao'], histtype='bar', align='mid', orientation='vertical')
ax2.set_title('hujintao' + str(np.where(dist_diff['hujintao'] >= np.max(dist_diff['hujintao']))[0]))
ax3 = plt.subplot(2,4,3)
plt.hist(dist_diff['jiangzemin'], histtype='bar', align='mid', orientation='vertical' )
ax3.set_title('jiangzemin'+str(np.where(dist_diff['jiangzemin'] >= np.max(dist_diff['jiangzemin'] ))[0]))
ax4 = plt.subplot(2,4,4)
plt.hist(dist_diff['dengxiaoping'], histtype='bar', align='mid', orientation='vertical' )
ax4.set_title('dengxiaoping'+str(np.where(dist_diff['dengxiaoping'] >= np.max(dist_diff['dengxiaoping'] ))[0]))
ax5 = plt.subplot(2,4,5)
plt.hist(dist_diff['wenjiabao'], histtype='bar', align='mid', orientation='vertical' )
ax5.set_title('wenjiabao' + str(np.where(dist_diff['wenjiabao'] >= np.max(dist_diff['wenjiabao'] ))[0]))
ax6 = plt.subplot(2,4,6)
plt.hist(dist_diff['maozedong'], histtype='bar', align='mid', orientation='vertical' )
ax6.set_title('maozedong' + str(np.where(dist_diff['maozedong'] >= np.max(dist_diff['maozedong'] ))[0]))
ax7 = plt.subplot(2,4,7)
plt.hist(dist_diff['zhouenlai'], histtype='bar', align='mid', orientation='vertical' )
ax7.set_title('zhouenlai' + str(np.where(dist_diff['zhouenlai'] >= np.max(dist_diff['zhouenlai'] ))[0]))
plt.show()
