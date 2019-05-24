import numpy as np
from sklearn.cluster import KMeans
import os

def get_index_every_nth(given_list, n):
    """
    :param given_list:
    :param n: every nth element gap
    :return: sub_index, gap boundary index
    for example:
    a = [4 2 3 9 6 6 8 5]
    b = get_index_every_nth(a, 3)
    :return
    c = [0,2,3,5,6,7]
    """
    num_list = len(given_list)
    mod = divmod(num_list, n)
    diviser = mod[0]
    res = mod[1]
    re = [-1]
    for i in np.arange(diviser):
        i += 1
        re.append(i*n-1)
    if res != 0:
        re.append(num_list-1)
    return re


def test_if_function():
    """
    This function tells us judgement do no loop
    which is different from matlab
    :return:
    """
    i = 0
    n = 10
    if i < n:
        print('current i is {}'.format(i))
        i += 1


def calculate_distance_matrix(matrix):
    """
    :param matrix: the input feature ebedding matrix, each row present the feature vector of one identity people's face,
                   and cols present the different faces
    :return: coorelation matrix
    """
    n, m = np.shape(matrix)
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            dist[i, j] = np.sqrt(np.sum(np.square(np.subtract(matrix[i, :], matrix[j, :]))))

    return dist


def cluster_not_same_class_image(embedding_feature):
    """
    :param embedding_matrix: the embedding feature matrix
    :return: True or False
    """
    kmeans = KMeans(n_clusters=2, random_state=0).fix(embedding_feature)

    return kmeans

def test_np_index(a):
    """
    This function tells how to index np.arange correctly
    So array and vector index comes from 0
    and like [   ), means including the right boundary without the left one
    :return:
    """
    # a = np.array([1, 2, 6], [9, 8, 3], [6, 8, 4])
    print(a)          # return [0, 1, 2, ... , 8]
    print(a[0:3])     # return [0, 1, 2]
    print(a[8:20])    # return [8]
    with open(os.path.join(os.getcwd(), 'average_emb.txt'), 'wb') as f:
        np.savetxt(f, a, fmt='%1.6f')
        # np.savetxt(f, 'average embedding')

if __name__ == '__main__':
    # print('return is {}'.format(get_index_every_nth(np.arange(9), 6)))
    # test_if_function()
    # test_np_index()
    matrix = np.array([[1.863,2.465,3.344,6.234,5.146],[2.123,5.157,3.263,6.657,4.612],[4.132,6.165,3.000,2.335,1.150]],dtype=np.float64)
    print(type(matrix))
    test_np_index(matrix)
    # dist = calculate_distance_matrix(matrix)
    # kmeans = cluster_not_same_class_image(matrix)
    # print(kmeans)