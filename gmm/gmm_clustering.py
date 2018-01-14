# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 22:10:55 2018

Soft clustering with GMM and Bayesian GMM
  http://scikit-learn.org/stable/modules/mixture.html

How expectation maximisation works
  https://www.youtube.com/watch?v=REypj2sy_5U

Molecular data as example

@author: mhaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


def getData():
    # Molecular data from  
    path_ftp = 'ftp://ftp.ics.uci.edu/pub/baldig/learning/Bergstrom/Bergstrom.xls'
    path_local = 'C:\Python34\\datasets\\Bergstrom.xls'

    # Read the excel file
    path = path_local
    df = pd.read_excel(path, sheetname=1, header=None)

    # Rename the second column
    df = df.rename(columns = {2: 'bp'})   # rename column '2' as 'bp', boiling point

    return(df)


def addFeatures(df):
    """
    In this function we add new features to the data table
    We encode the SMILES formula to some values
    These new features are added as new columns to the table
    - number of C, O, N in the formula
    - relative values: n[O]/n[C], n[N]/n[C], (n[N]+n[O])/n[C] in the formula
    You can also make yourself new features following the examples below
    """
    # Go through the dataframe line by line
    # Calculate how many elements there are in the formula
    # and insert new calculated attributes into new columns
    # Note: you can also make yourself new columns
    
    # print(addFeatures.__doc__)
    debug=False
    ind=0
    for formula in df[1]:
        # Calculate how many O, N, C there are in the formula
        no=formula.count('O')
        nn=formula.count('N')
        nc=formula.count('C') + formula.count('c')
 
        # How many double bonds there are in the formula
        ndouble=formula.count('=')

        if(debug): print(ind, nc, formula)

        # Add new columns to the data table df
        # Add the amount of O, N...
        df.set_value(ind,'n_o', no)
        df.set_value(ind,'n_n', nn)
        df.set_value(ind,'n_c', nc)
        df.set_value(ind,'n_dbl', ndouble)    

        # Add the relative amounts: number of O / number of C etc.
        df.set_value(ind,'n_o_c', no/nc)
        df.set_value(ind,'n_n_c', nn/nc)
        df.set_value(ind,'n_dbl_c', ndouble/nc)

        # Relative amount of N+O elements / number of carbons
        df.set_value(ind,'n_no_c', (nn+no)/nc)
        ind+=1
    return(df)


def make_ellipses(gmm, ax, ncomp, Y_):
    # Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
    # License: BSD 3 clause
    # 
    # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    # Minor modification by Mikko Hakala
    import matplotlib as mpl
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'blue', 'brown']

    for n, color in enumerate(colors[0:ncomp]):
        # For Bayesian fitting, we must get out of the loop if there
        # are no predictions to a class
        if(np.any(Y_ == n)):
            print('n:', n)
        else:
            print('n:', n, 'label not found in Y_ - skipping this ellipsis')
            continue
        
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def make_ellipses_explained():
    """ 
    Explanation of the make_ellipses function 
    TODO
    """
    pass



def gaussian_mixture_model(X, n_components, cov_type):
    # Plot the data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(X[:, 0], X[:, 1])   

    # Fit the model
    gmm=GaussianMixture(n_components=n_components, covariance_type=cov_type)
    gmm.fit(X)
    print('n_comp, BIC:', n_components, gmm.bic(X))

    # Get the labels
    Y_ = gmm.predict(X)

    # Plot the model
    make_ellipses(gmm, ax, n_components, Y_)
    
    plt.show()

    return(gmm)


def bayesian_gaussian_mixture_model(X, n_components, cov_type):
    # Plot the data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(X[:, 0], X[:, 1])   

    # Fit the model
    # bgmm=BayesianGaussianMixture(n_components=n_components, covariance_type=cov_type, max_iter = 500)
    wg_c_prior = 0.000001* 1./n_components  # smaller: finds less components
    # wg_c_prior = 100.0  # bigger: tends to find more components 
    bgmm=BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=wg_c_prior, covariance_type=cov_type, max_iter = 500)
    bgmm.fit(X)

    # Get the labels
    Y_ = bgmm.predict(X)
    print('min/max label Y_ :', np.min(Y_), np.max(Y_))
    # print('Y_:', Y_)

    # Report the unique lables
    print('Unique Y_ labels:', np.unique(Y_))
    y = np.bincount(Y_)
    ii = np.nonzero(y)[0]
    print('label:', ii)
    print('count:', y[ii]) 

    # Plot the model
    make_ellipses(bgmm, ax, n_components, Y_)
    
    # Generate and plot random samples
    if(False):
        n=200
        genx, labelx = bgmm.sample(n)
        colors = ['navy', 'turquoise', 'darkorange', 'red', 'yellow', 'blue', 'brown']
        for i in range(genx.shape[0]):
            ax.scatter(genx[i, 0], genx[i, 1], color = colors[labelx[i]])     
    
    plt.show()

    return(bgmm)



def main():

    # Get the data    
    df = getData()    
    df = addFeatures(df)
    
    # Examine
    print(df.head(3))
    print(df.describe())   # Summary statistics

    # Study 2-D data
    #features = ['n_c', 'bp']
    features = ['bp', 'n_no_c']
    #features = ['n_no_c', 'bp']
    X = df[features].values


    print('-- Gaussian Mixture Model --')

    if(False):
        for comp in range(4):
            model = gaussian_mixture_model(X, n_components=comp+1, cov_type='full')

    print('---')

    if(False):
        # Study 3-D data 
        features = ['n_c', 'bp', 'n_no_c']
        X = df[features].values
        for comp in range(2):
            model = GaussianMixture(n_components=comp+1, covariance_type='full')
            model.fit(X)
        print('means:', model.means_)
        
        
    if(False):
        # Print data if needed
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.scatter(X[:, 1], X[:, 2])   
            
    #input("\nPress Enter to continue")


    print('-- Bayesian Gaussian Mixture Model --')

    features = ['bp', 'n_no_c']
    X = df[features].values

    if(True):
        ncomp = 5
        print('# trial components:', ncomp)
        bayesian_gaussian_mixture_model(X, n_components=ncomp, cov_type='full')



    print("Done")
    
    
if __name__ == '__main__':
    main()
