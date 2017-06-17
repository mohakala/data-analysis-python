# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:26:57 2017

Use various ML algorithms to predict the price of an apartment

@author: mikko hakala
"""

import numpy as np
#import pandas as pd

import sys
sys.path.insert(0, 'C:\Python34')

from mlproject import mlproject as mlp

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def study_knn(ml):
    from sklearn import neighbors
    # weights = 'uniform'     #  'uniform' or 'distance'
    weights = 'distance'     #  'uniform' or 'distance'
    for n_neighbors in (range(1, 7)):
        print('\n\nneighbors:', n_neighbors)
        model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        ml.score(model)
    print('found best CV w/ 5 neighbors, dist,. criterion = 71.9 (better) and uniform')


def study_linreg(ml):
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    ml.score(model, iprint=4)
    ml.print_coef()


def study_decisionTree(ml):
    from sklearn import tree
    model = tree.DecisionTreeRegressor()
    ml.score(model)
    ml.print_coef()


def study_randomForest(ml):
    from sklearn import ensemble 
    model = ensemble.RandomForestRegressor()
    ml.score(model)
    # ml.score_print()


def study_standard_ml(ml):
    # Set features and target 
    features = ['Huoneet', 'm2', 'Rv']
    #features = ['m2']
    target = ['Vh']
    allfeatures = target + features

    # Set indices for train, validate, test split
    # - total data length: 326
    ind = [234, 294]
    ml.set_xy(target, features, ind)


    print("*K nearest neighbors")
    study_knn(ml)
    
    print("*Linear regression")
    study_linreg(ml)
    
    print("*Decision tree")
    study_decisionTree(ml)

#    Some problems with random forest
#    print("Random Forest")
#    study_randomForest(ml)

    # Test score for knn
    print("**Test score for decision tree")
    from sklearn import neighbors
    n_neighbors=5
    model = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    ml.score(model, iprint=0)
    ml.score_print(printTestScore=True)


def study_nn(ml):
    features = ['Huoneet', 'm2', 'Rv']
    #features = ['m2']
    target = ['Vh']
    allfeatures = target + features

    #print('X:', ml.Xtest[:2])
    #print('y:', ml.ytest[:2])

    # Maximum values of all features since data needs to be scaled by max
    max_values = ml.df[allfeatures].max(axis=0)
    MaxVh = max_values[0] 
    print('Max of features:', max_values)
    print('Max target:', MaxVh)

    # Scale all data with max (for NN studies) to be betw 0 and 1
    ml.df[allfeatures] = ml.df[allfeatures] / max_values
    print(ml.df.head())

    # Set indices for train, validate, test split
    # - total data length: 326
    ind = [234, 294]
    ml.set_xy(target, features, ind)
    
    # Some sizes of vectors (features and targets) 
    A = ml.Xtest.shape[1]
    B = ml.ytest.shape[1]
    
    # Placeholders and variables
    X = tf.placeholder(tf.float32, [None, A])
    Y_= tf.placeholder(tf.float32, [None, B])


    # Normal linear regression gives these values:
    # .. intercept: [-2435988.292]
    # .. params:  [[ 20489.955   1260.256   1232.52 ]]

    ## A. Single-layer perceptron (slp) 
    make_slp = False
    if(make_slp):    
        W = tf.Variable(tf.zeros([A, B]))
        # W from ordinary linear fit
        #Wvar = np.float32(np.array([ 20489.955, 1260.256, 1232.52 ])).reshape(-1, 1)
        #W = tf.Variable( Wvar )
        b = tf.Variable(tf.zeros([B]))
        #bvar = np.float32(np.array([-2435988.292]))
        #b = tf.Variable(bvar)
        # Model
        # Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, A]), W) +b)
        Y = tf.matmul(tf.reshape(X, [-1, A]), W) + b


    ## B. Multilayer perceptron
    make_mlp = False
    if(make_mlp):
        print("--Note: As the network is now, unstable test set, maybe overfits")
        print("--Note: Try dropout etc.")
        # Variables
        A = A
        K = 200
        L = 100
        M = 60
        N = 30
        O = B
        W1 = tf.Variable(tf.truncated_normal([A, K], stddev=0.1))
        B1 = tf.Variable(tf.zeros([K]))
        W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
        B2 = tf.Variable(tf.zeros([L]))
        W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        B3 = tf.Variable(tf.zeros([M]))
        W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        B4 = tf.Variable(tf.zeros([N]))
        W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        B5 = tf.Variable(tf.zeros([O]))
        # Model
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
        #Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
        pkeep = 0.75
        Y5 = tf.nn.dropout(Y4, pkeep)
        Y = tf.matmul(Y5, W5) + B5

    ## C. Light multilayer perceptron
    make_mlp = True
    if(make_mlp):
        print('Lightweight MLP')
        # Variables
        A = A
        K = 200
        L = 100
        M = 60
        N = 30
        O = B
        W1 = tf.Variable(tf.truncated_normal([A, K], stddev=0.1))
        B1 = tf.Variable(tf.zeros([K]))
        W2 = tf.Variable(tf.truncated_normal([K, N], stddev=0.1))
        B2 = tf.Variable(tf.zeros([N]))
        W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        B5 = tf.Variable(tf.zeros([O]))
        # Model
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        pkeep = 0.75
        Y5 = tf.nn.dropout(Y2, pkeep)
        Y = tf.matmul(Y5, W5) + B5



    batch_size = 234
    
    # https://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
    # Loss function
    # loss = tf.reduce_sum( (Y_ - Y)*(Y_ - Y) )
    # loss = tf.reduce_mean(tf.square(tf.subtract(Y_, Y))) # probably ok
    loss = tf.reduce_sum(tf.pow(Y_-Y, 2))/(2*batch_size)

    # Accuracies etc.
    rmse  = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y_, Y)))
    ssres = tf.reduce_sum(tf.squared_difference(Y_, Y))
    sstot = tf.reduce_sum(tf.squared_difference(Y_, tf.reduce_mean(Y_)))
    
    # Training step and optimizer
    learning_rate = 0.0002   # was: 0.005
    # To try: learning rate decay
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    print("--Note: AdamOptimizer much faster than GradientDescent!")

    
    # Initialize
    init = tf.global_variables_initializer()



    def r2_value(ssres, sstot):
        return(1.0 - (ssres/sstot))

    def iterate_minibatches(inputs, targets, batch_size):
        assert len(inputs) == len(targets)
        N = len(inputs)
        indices = np.arange(N)
        np.random.shuffle(indices)
        for i in range(0, N - batch_size + 1, batch_size):
            indices_batch = indices[i: i + batch_size]
            yield inputs[indices_batch], targets[indices_batch]

    # Training loop
    num_steps=30000

    # Store data on RMSE accuracy and loss
    metrics = []

    sess = tf.Session()
    sess.run(init)
    for i in range(num_steps):
        for batch in iterate_minibatches(ml.Xtrain, ml.ytrain, batch_size):
            inputs, targets = batch
            train_data={X: inputs, Y_: targets}
            sess.run(train_step, feed_dict=train_data)

            
        # Print accuracy 
        if(i%500==0 and i>0):
            # Training set
            acc = sess.run(rmse, feed_dict=train_data)
            ssres_, sstot_ = sess.run([ssres, sstot], feed_dict=train_data)
            r2 = r2_value(ssres_, sstot_)
            test_data = {X: ml.Xtest, Y_: ml.ytest}
            
            # Test set
            acc_test, loss_test = sess.run([rmse, loss], feed_dict=test_data)
            ssres_, sstot_ = sess.run([ssres, sstot], feed_dict=test_data)
            r2_test = r2_value(ssres_, sstot_)

            metrics.append([i, r2, r2_test, acc_test*max_values[0]])

            # Print variables during training
            #w_ = sess.run(W)
            #b_ = sess.run(b)
            #wreal = w_ * max_values[1:4].reshape(-1, 1)
            #print('W, b, r, r2_test2', wreal.reshape(1, -1), b_, r2, r2_test)
            #print(i, 'Acc (rmse, r2):', acc*MaxVh, r2, ' Test data (rmse, loss):', acc_test*MaxVh, loss_test)
            print('i, r2, r2_test, rmse_test', i, r2, r2_test, acc_test*max_values[0])
            

    # Plot metrics
    metrics=np.array(metrics)
    print(metrics)
    ml.plot(metrics[:, 0], metrics[:, 1], metrics[:, 0], metrics[:, 2])
    ml.plot(metrics[:, 0], metrics[:, 3])

    # Finally, rescale all data back to original values 
    print('*Rescaling the features back to original values')
    ml.df[allfeatures] = ml.df[allfeatures] * max_values
    ind = [234, 294]
    ml.set_xy(target, features, ind)
    #print(ml.df.head())

    #assert False
    


def main():    
    # Load data
    ml = mlp.mlproject()
    path = 'C:/Python34/datasets/nurmijarvi_asunnot_250316.csv'
    ml.getData(path)

    print('Towns in the dataset:')
    print(ml.df['Kaupunginosa'].value_counts())
    print('---')
    
    print('Max values:')
    print(ml.df.max(axis=0))
    print('---')

    print('Missing values in columns:\n', ml.missingValues())
    print('Filling missing value in Kaupunginosa with the most popular one')
    ml.fillMissingCategorical('Kaupunginosa')
    print('---')

    print('Randomizing the rows of the dataframe')
    ml.randomizeRows(0)

    print(ml.df.head(3))
    print('Missing values in columns:\n', ml.missingValues())
    ml.examine()    

    # TODO: Encode kaup.osa into features
    # Max Vh
    max_y = np.max(ml.df['Vh'])
    print('max Vh:', max_y, 'euro')

    # Correlation matrix
    print('Correlation matrix')
    # print(ml.df[allfeatures].corr())
    print(ml.df.corr())

    # Study standard ML models
    print("\n*Study standard ML models")
    study_standard_ml(ml)

    # Study neural network
    print("---------------------------------------------")
    print("\n*Study simple neural network for regression")
    study_nn(ml)



    print('------------\nDone')




if __name__ == '__main__':
    main()