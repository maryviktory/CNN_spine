import numpy as np
from sklearn.cluster import k_means


def k_means_clustering(ndarray, n_clusters):
    flat_vol = np.expand_dims(ndarray.flatten(), axis=1)
    _, clusters, _ = k_means(n_clusters=n_clusters, random_state=0, X=flat_vol)
    reshaped = np.reshape(clusters, ndarray.shape)
    return reshaped

