# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:37:54 2019

https://machinelearningmastery.com/expectation-maximization-em-algorithm/

@author: mikko
"""

# example of a bimodal constructed from two gaussian processes
from numpy import hstack
from numpy.random import normal
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

def example1():
    twoDim = False
    
    # generate a sample
    X1 = normal(loc=20, scale=2, size=300)
    X2 = normal(loc=40, scale=5, size=700)
    X = hstack((X1, X2)).reshape(-1, 1)

    if twoDim:    
        XB = normal(loc = 0, scale=1, size = X.shape[0]).reshape(-1, 1)
        X = np.concatenate((X, XB), axis=1)


    # plot the histogram
    plt.hist(X[:, 0], bins=50, density=True)
    plt.show()
    if twoDim:    
        plt.hist(X[:, 1], bins=50, density=True)
        plt.show()
        plt.scatter(X[:, 0], X[:, 1], s=40, cmap='viridis');
        plt.show()
    """
    If the number of processes was not known, a range of different 
    numbers of components could be tested and the model with the best 
    fit could be chosen, where models could be evaluated using scores 
    such as Akaike or Bayesian Information Criterion (AIC or BIC).
    """

    # fit model
    model = GaussianMixture(n_components=2, init_params='kmeans').fit(X)
    #model.fit(X)
    # predict latent values
    yhat = model.predict(X)
    
    if twoDim: 
        plt.scatter(X[:, 0], X[:, 1], c=yhat, s=40, cmap='viridis')
        plt.show()

    
    print('weights: ', model.weights_)
    print('means: ', model.means_)
    print('covariances: ', model.covariances_)
    print(model.lower_bound_) 
    print(model.n_iter_)
    

    # check latent value for first few points
    print(yhat[:50])
    print(X[:20].flatten())
    # check latent value for last few points
    print(yhat[-50:])
    print(X[-20:].flatten())


    print('params', model.get_params())
    
    
    
def example2():
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    # Generate some data
    from sklearn.datasets.samples_generator import make_blobs
    X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting    
    
    
    gmm = GaussianMixture(n_components=4).fit(X)
    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    plt.plot()
    
    print('weights: ', gmm.weights_)
    print('means: ', gmm.means_)
    print('covariances: ', gmm.covariances_)
    print(gmm.lower_bound_) 
    print(gmm.n_iter_)





def main():
    example1()
    #example2()
    
if __name__ == '__main__':
    main()