"""
This script cluster single dimension data by MeanShift method
"""
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

def cluster_indice(scores):

    scores = np.reshape(scores, (-1, 1))
    ms = MeanShift(bandwidth=None, bin_seeding=True)
    ms.fit(scores)
    labels_tmp = ms.labels_
    labels = [1 if i > 0 else i for i in labels_tmp]
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    labels = np.asarray(labels)
    face_indice = np.squeeze(np.where(labels == 1))

    return face_indice