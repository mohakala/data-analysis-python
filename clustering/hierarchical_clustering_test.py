# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:33:31 2018

@author: mhaa
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram


distMatrix = np.array([[0, 2, 5], [2, 0, 6], [5, 6, 0]])

# https://stats.stackexchange.com/questions/2717/clustering-with-a-distance-matrix



# https://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
import scipy.spatial.distance as ssd
# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(distMatrix) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j

Z = scipy.cluster.hierarchy.linkage(distArray, method='single', metric='euclidean')

fig = plt.figure(figsize=(5, 5))
dn = dendrogram(Z)


#>>> from scipy.cluster.hierarchy import dendrogram, linkage
#>>> from matplotlib import pyplot as plt
#>>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

#>>> Z = linkage(X, 'ward')
#>>> fig = plt.figure(figsize=(25, 10))
#>>> dn = dendrogram(Z)

#>>> Z = linkage(X, 'single')
#>>> fig = plt.figure(figsize=(25, 10))
#>>> dn = dendrogram(Z)
#>>> plt.show()

