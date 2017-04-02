# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:26:57 2017

@author: mhaa
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
    ml.fillMissing('Kaupunginosa')
    ml.randomizeRows()

    print(ml.df.head())
    print('Missing values in columns:\n', ml.missingValues())
    ml.examine()    


        
    features = ['Huoneet', 'm2', 'Rv']
    target = 'Vh'
    # Set indices for train, validate, test split
    ind = [234, 294]
    ml.set_new(target, features, ind)
    print('Correlation matrix')
    print(ml.df[features].corr())


    from sklearn import neighbors
    weights = 'uniform'     #  'uniform' or 'distance'
    n_neighbors = 3
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    ml.score(knn)
    ml.score_print()
    
    
    from sklearn import linear_model
    linmod = linear_model.LinearRegression()
    ml.score(linmod)
    ml.score_print()




    print('------------')
    print('Done')




if __name__ == '__main__':
    main()