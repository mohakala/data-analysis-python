# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:26:57 2017

@author: mikko hakala
"""

#import numpy as np
#import pandas as pd

import sys
sys.path.insert(0, 'C:\Python34')

from mlproject import mlproject as mlp





def main():
    
    ml = mlp.mlproject()
    path = 'C:/Python34/datasets/nurmijarvi_asunnot_250316.csv'
    ml.getData(path)

    print('Missing values in columns:\n', ml.missingValues())
    ml.fillMissingCategorical('Kaupunginosa')
    ml.randomizeRows(0)

    print(ml.df.head())
    print('Missing values in columns:\n', ml.missingValues())
    ml.examine()    

    # Set features and target 
    features = ['Huoneet', 'm2', 'Rv']
    target = ['Vh']
    allfeatures = target + features

    # Set indices for train, validate, test split
    # - total data length: 326
    ind = [234, 294]
    ml.set_new(target, features, ind)

    print('Correlation matrix')
    print(ml.df[allfeatures].corr())


    # Train, validate, test various models
    
    


    from sklearn import neighbors
    # weights = 'uniform'     #  'uniform' or 'distance'
    weights = 'distance'     #  'uniform' or 'distance'
    for n_neighbors in (range(1, 8)):
        print('\n\nneighbors:', n_neighbors)
        model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        ml.score(model)
        ml.score_print()
    print('found best CV w/ 5 neighbors, dist,. criterion = 71.9 (better) and uniform')
    
    
    
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    ml.score(model)
    ml.score_print()
    ml.print_coef()


    from sklearn import tree
    model = tree.DecisionTreeRegressor()
    ml.score(model)
    ml.score_print()
    ml.print_coef()


    from sklearn import ensemble 
    model = ensemble.RandomForestRegressor()
    ml.score(model)
    ml.score_print()


    





    print('------------\nDone')




if __name__ == '__main__':
    main()