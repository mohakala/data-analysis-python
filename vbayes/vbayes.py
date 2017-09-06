# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:35:54 2017

From https://arxiv.org/pdf/1410.0870.pdf
http://bayespy.org/examples/gmm.html
"""

import os
cwd = os.getcwd()
print(cwd)

import  numpy as np

# Two 2D-Gaussians at (0,0) and (2,2)
N = 500; D = 2
data = np.random.randn(N, D)
data [:200 ,:] += 2*np.ones(D)

# Maximum number of clusters
K = 5


from  bayespy  import  nodes

# Initialize the means mu of the unknown clusters
mu = nodes.Gaussian(np.zeros(D), 0.01* np.identity(D), plates =(K,))
print(mu)
print("np.identity(D):", np.identity(D))

# Initialize the precisions Lambda of the unknown clusters
#   Wishart distribution for conjugate prior
Lambda = nodes.Wishart(D, D*np.identity(D), plates =(K,))
print(Lambda)

# Cluster probabilities: Dirichlet prior
alpha = nodes.Dirichlet (0.01* np.ones(K))

# Cluster assignments 
z = nodes.Categorical(alpha, plates =(N,))

# Observations from Gaussian mixture distribution
y = nodes.Mixture(z, nodes.Gaussian , mu , Lambda)

# Observations (data)
y.observe(data)

# Variational Bayesian inference engine
from  bayespy.inference  import  VB
Q = VB(y, mu , z, Lambda , alpha)

# Random initial assignments
z.initialize_from_random ()

Q.update(repeat =200)

import  bayespy.plot as bpplt
bpplt.gaussian_mixture_2d(y, alpha=alpha)
bpplt.pyplot.show()

# Print the parameters of the approximate posterior distributions
print(alpha)



