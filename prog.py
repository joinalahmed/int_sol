import numpy as np
from sklearn.cluster import KMeans
import os
import sys

def kmeans_cluster(data_file,n_c):
    data= np.loadtxt(data_file)
    kmeans = KMeans(n_clusters=n_c, random_state=0).fit(data)
    return kmeans.cluster_centers_

if __name__ == '__main__':
    data_file = sys.argv[1]
    cluster = int(sys.argv[2])
    x=kmeans_cluster(data_file,cluster)
    np.savetxt('clusters.txt', x, delimiter=',')

