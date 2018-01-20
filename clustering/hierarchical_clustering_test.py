# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:33:31 2018

@author: mhaa
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram


# distMatrix = np.array([[0, 2, 5], [2, 0, 6], [5, 6, 0]])

path_local = 'C:\Python34\\datasets\\kaupungit_etaisyysmatriisi.csv'
# df = pd.read_csv(path_local, header=None)
distMatrix = np.genfromtxt(path_local, delimiter=',')
print(distMatrix.shape)


kaupunki = [
"Espoo",
"Helsinki",
"Joensuu",
"Jyväskylä",
"Kotka",
"Kuopio",
"Lahti",
"Lappeenranta",
"Oulu",
"Pori",
"Rovaniemi",
"Tampere",
"Turku",
"Vaasa",
"Vantaa"
]


print('==== Scipy ==========')


# https://stats.stackexchange.com/questions/2717/clustering-with-a-distance-matrix

# https://stackoverflow.com/questions/18952587/use-distance-matrix-in-scipy-cluster-hierarchy-linkage
import scipy.spatial.distance as ssd
# convert the redundant n*n square matrix form into a condensed nC2 array
distArray = ssd.squareform(distMatrix) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
print(distArray.shape)

#method = 'ward'
method = 'average'
Z = scipy.cluster.hierarchy.linkage(distArray, method=method, metric='euclidean')
# method = 'single'

fig = plt.figure(figsize=(5, 5))
dn = dendrogram(Z)
plt.show()

""" 
Explanation of Z

A by 4 matrix Z is returned. At the i:th iteration, clusters with 
indices Z[i, 0] and Z[i, 1] are combined to form cluster. A 
cluster with an index less than corresponds to one of the original 
observations. The distance between clusters Z[i, 0] and Z[i, 1] is 
given by Z[i, 2]. The fourth value Z[i, 3] represents the number 
of original observations in the newly formed cluster.
"""

print('\nFar from the rest')
for i in (8, 10):
    print(i, kaupunki[i])

print('\n---- Results ----')
list1 = (8, 10)
list2 = (3, 2, 5)
list3 = (0, 1, 14, 7, 4, 6)
list4 = (13, 12, 9, 11)

for index, j in enumerate((list1, list2, list3, list4)):
    print('\nCluster', index)
    for i in j:
        print(i, kaupunki[i])


print('\n==== Scikit-learn ============')

n_clusters = 5
linkage = 'average'
affinity = 'precomputed'
model = AgglomerativeClustering(n_clusters=n_clusters, 
                                affinity=affinity, linkage=linkage)
model.fit(distMatrix)

print('City, cluster')
for index, label in enumerate(model.labels_):
    print(kaupunki[index], label)
    

#>>> from scipy.cluster.hierarchy import dendrogram, linkage
#>>> from matplotlib import pyplot as plt
#>>> X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

#>>> Z = linkage(X, 'ward')
#>>> fig = plt.figure(figsize=(25, 10))
#>>> dn = dendrogram(Z)


