""" K-Means clustering
http://spark.apache.org/docs/latest/mllib-clustering.html#k-means
http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.KMeans

* kmeansOned(fileName):
Read the simple data
Build the model 
Evaluate the WSSSE
Do a couple of predictions

* kmeansMultid(fileName):
Read a multicolumn data
Build the model 
Evaluate the WSSSE
Do a couple of predictions

"""

from pyspark import SparkContext
import sys
import psutil

from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import numpy as np

from pyspark.mllib.linalg import Vectors

def kmeansOned(fileName,nClusters):
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    sc = SparkContext("local")
    data=sc.textFile(fileName)
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
    clusters = KMeans.train(parsedData, nClusters, maxIterations=10, initializationMode="random")

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))
    return()

def kmeansExample1():
    # http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.clustering.KMeans
    data = array([0.0,0.0, 1.0,1.0, 9.0,8.0, 8.0,9.0]).reshape(4, 2)
    sc = SparkContext("local")
    model = KMeans.train(sc.parallelize(data), 2, maxIterations=10, initializationMode="random",seed=50, initializationSteps=5, epsilon=1e-4)
    print('Number of clusters:',model.k)
    print('Index for first item:',model.predict(array([0.0,0.0])))
    print('Index for last item:',model.predict(array([8.0,9.0])))
    print('Index for 8.0 8.0:',model.predict(array([8.0,8.0])))
    print('Index for -1.0 -4.0:',model.predict(array([-1.0,-4.0])))
    print('Test:',model.predict(array([0.0, 0.0])) == model.predict(array([1.0, 1.0])))
    return()

def kmeansMultid(data,nClusters):
    # Multicolumnar data

    # To do: Now data is passed simply as an argument and parallelized.
    # Add the option to read from file.

    sc = SparkContext("local")
    model = KMeans.train(sc.parallelize(data), nClusters, maxIterations=10, initializationMode="random",seed=50, initializationSteps=5, epsilon=1e-4)
    print('Number of clusters:',model.k)
    print('Number of instances for training:',len(data))
    
    # Predictions
    case=array([4,2,116,200000,1724,1978])
    print('Prediction for:',case,' is:',model.predict(case))
    case=array([4,0,84,210000,2500,2003])
    print('Prediction for:',case,' is:',model.predict(case))
    case=array([4,1,226,440000,1947,1927])
    print('Prediction for:',case,' is:',model.predict(case))
    
      
if __name__ == '__main__':

    # Test 1
    fileName="data/mllib/kmeans_data.txt"
    nClusters=2 # Number of clusters for K-Means
    if(False): kmeansOned(fileName,nClusters)

    # Test 2
    if(False): kmeansExample1()

    # Test 3 kmeansMultid
    if(True):
        df=np.loadtxt('tests_mh/asunnot_250316_cleaned_numerical.csv',delimiter=',')
        # Take 80% of values for training set
        ndf=len(df)
        upperlimit=int(0.80*ndf) 
        trainRange=np.arange(0,upperlimit)
        trainingSet=df[trainRange,:]
        # Do clustering
        kmeansMultid(trainingSet,10)

    # Other tests
    if(False):
        print(Vectors.sparse(100, [0, 2], [1.0, 3.0]))









