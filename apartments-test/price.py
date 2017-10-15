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


def study_standard_ml(ml):
    features = ['Huoneet', 'm2', 'Rv']
    target = ['Vh']
    # allfeatures = target + features

    # Set indices for train, validate, test split
    # - total data length: 326
    ind = [234, 294]
    ml.set_xy(target, features, ind)

    print("\n*K nearest neighbors")
    study_knn(ml)
    
    print("\n*Linear regression")
    study_linreg(ml)
    
    print("\n*Decision tree")
    study_decisionTree(ml)

    print("\n*Skipping Random Forest. To do: fix the warnings") 
    # study_randomForest(ml)



def study_knn(ml):
    from sklearn import neighbors

    print_results = False
    if(print_results):
        iprint=1
    else:
        iprint=0

    weights = 'distance'
    for n_neighbors in (range(3, 16)):
        if(print_results):
            print('\nneighbors:', n_neighbors, 'weights:', weights)
        model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        ml.score(model, iprint=iprint)
    print('\nWeights =', weights)
    print('Best CV w/ 11 neighbors:')
    model = neighbors.KNeighborsRegressor(n_neighbors=11, weights=weights)
    ml.score(model, iprint=4)
    ml.score_print(printTestScore=True)

    
    weights = 'uniform'
    for n_neighbors in (range(3, 11)):
        if(print_results):
            print('\nneighbors:', n_neighbors, 'weights:', weights)
        model = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        ml.score(model, iprint=iprint)
    print('\nWeights =', weights)
    print('Best CV w/ 5 neighbors:')
    model = neighbors.KNeighborsRegressor(n_neighbors=5, weights=weights)
    ml.score(model, iprint=0)
    ml.score_print(printTestScore=True)


def study_linreg(ml):
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    ml.score(model, iprint=4)
    ml.print_coef()
    ml.score_print(printTestScore=True)


def study_decisionTree(ml):
    from sklearn import tree
    model = tree.DecisionTreeRegressor()
    ml.score(model, iprint=0)
    ml.print_coef()
    ml.score_print(printTestScore=True)


def study_randomForest(ml):
    from sklearn import ensemble 
    model = ensemble.RandomForestRegressor()
    ml.score(model)
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
    print('ml.Xtest.shape:', ml.Xtest.shape)
    
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
        pkeep = 0.80
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

        #W1 = tf.Variable(tf.truncated_normal([A, K], stddev=0.1))
        W1 = tf.get_variable("W1", shape=[A, K],
           initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        B1 = tf.Variable(tf.zeros([K]))
        #W2 = tf.Variable(tf.truncated_normal([K, N], stddev=0.1))
        W2 = tf.get_variable("W2", shape=[K, N],
           initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        B2 = tf.Variable(tf.zeros([N]))
        #W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        W5 = tf.get_variable("W5", shape=[N, O],
           initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        B5 = tf.Variable(tf.zeros([O]))
        # Model
        Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
        pkeep = 1.0
        Y5 = tf.nn.dropout(Y2, pkeep)
        Y = tf.matmul(Y5, W5) + B5


    # Batch size
    # batch_size = np.int(ml.Xtest.shape[0]/2.0)
    batch_size = 234   # total train size
    #batch_size = 117
    print('Batch size:', batch_size)
    
    # https://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
    # Loss function
    # loss = tf.reduce_sum( (Y_ - Y)*(Y_ - Y) )
    # loss = tf.reduce_mean(tf.square(tf.subtract(Y_, Y))) # probably ok
    loss = tf.reduce_sum(tf.pow(Y_-Y, 2))/(2*batch_size)

    # Accuracies etc.
    rmse  = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y_, Y)))
    ssres = tf.reduce_sum(tf.squared_difference(Y_, Y))
    sstot = tf.reduce_sum(tf.squared_difference(Y_, tf.reduce_mean(Y_)))
    mae = tf.reduce_mean(tf.abs(Y_- Y))


    # Number of epochs

    # num_steps=12730 # looks good for light MLP
    # num_steps=12350 # 
    num_steps=5000 
    
    
    # Optimizer
    
    """
    OPTIMIZER
    IG:
    Currently actively in use: SGD, SGD+mom, RMSProp, RMSProp+mom, AdaDelta, Adam
    Choice depends on user'f familiarity of algorithm and hyperpar tuning 
    Adam fairly robust to choice of hyperparameter
    """
    learning_rate = 0.005   # was: 0.005
    # To try: learning rate decay

    # RMSProp
    #train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    
    # AdaGrad
    #train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    
    # Gradient descent and with momentum
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #momentum_rate=0.9
    #train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate).minimize(loss)

    # Adam 
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #print("--Note: AdamOptimizer can be much faster than GradientDescent!")

    
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
        if(i%500==0 and i>0 or i==100 or i==num_steps):
            # Use all data points in training set
            train_data={X: ml.Xtrain, Y_: ml.ytrain}
            ssres_, sstot_ = sess.run([ssres, sstot], feed_dict=train_data)
            r2 = r2_value(ssres_, sstot_)
            
            # Validation set
            val_data = {X: ml.Xval, Y_: ml.yval}
            acc_val, loss_val = sess.run([rmse, loss], feed_dict=val_data)
            ssres_, sstot_ = sess.run([ssres, sstot], feed_dict=val_data)
            r2_val = r2_value(ssres_, sstot_)

            # Add to list results: index, r2_train, r2_test, rmse
            metrics.append([i, r2, r2_val, acc_val*MaxVh])

            # Print variables during training
            #w_ = sess.run(W)
            #b_ = sess.run(b)
            #wreal = w_ * max_values[1:4].reshape(-1, 1)
            #print('W, b, r, r2_test2', wreal.reshape(1, -1), b_, r2, r2_test)
            #print(i, 'Acc (rmse, r2):', acc*MaxVh, r2, ' Test data (rmse, loss):', acc_test*MaxVh, loss_test)
            print('i, r2, r2_val, rmse_test', i, r2, r2_val, acc_val*MaxVh)
            
    # Final round
    test_data = {X: ml.Xtest, Y_: ml.ytest}
    acc_test, mae_test = sess.run([rmse, mae], feed_dict=test_data)
    ssres_, sstot_ = sess.run([ssres, sstot], feed_dict=test_data)
    r2_test = r2_value(ssres_, sstot_)
    print('Final test data r2:', r2_test)
    print('RMSE:', acc_test*MaxVh, 'MAE:', mae_test*MaxVh)

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

def print_results_for_reference():
    print('Linear regression:')
    print('- R2 0.668, R2 valid 0.504, Cross-val 0.63+-0.19, R2 test 0.791')
    print('SLP the samem batch 234:')
    print('--> 17 k Adam(0.01) ok')
    print('--> 17 k Adam(0.3) is noisy')
    print('--> 40 k grad(0.3)+mom(0.9)')


    print('KNN w=distance, k=11:')
    print('- R2 0.998, R2 valid 0.74, Cross-val 0.73+-0.26, R2 test 0.82')

    print('Light multilayer perceptron, Adam, dropout 0.75:')
    print('- R2~0.70, R2 test 0.6-0.8')

    print('Light multilayer perceptron, learn rate 0.005, 250 k iter:')
    print('- R2~0.86 (max 0.90), R2 valid 0.59, R2 test 0.16')
    print('- valid.max at ~50 k')

    print('Light MLP, learn rate 0.005, 100 k iter, batchsz 234:')
    print('- R2 0.91, R2 valid 0.64, R2 test 0.39')
    print('- valid.max at ~40 k')

    print('MLP, learn rate 0.001, 50 k iter, Adam, batchsz 234:')
    print('- R2 0.93 (max 0.95), R2 valid 0.72, R2 test 0.58')
    print('- valid.max at ~12.4 k')
    print('-- this with 12.35 K iter: valid 0.69, test 0.72')

    print('MLP, learn rate 0.001, 50 k iter, Adam, batchsz 100, drop 0.8:')
    print('- R2 0.91, R2 valid 0.70, R2 test 0.74')




def main():    
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
    print('Missing values in Kaupunginosa:', sum( ml.df['Kaupunginosa'].isnull() ) )
    if(False):
        print('Rows with missing Kaupunginosa:')
        print(ml.df[ml.df['Kaupunginosa'].isnull()])

    method_to_fillna = 'random'
    print('Filling missing value in Kaupunginosa with:', method_to_fillna)
    ml.fillMissingCategorical('Kaupunginosa', method=method_to_fillna)
    print('---')


    print('Randomizing the rows of the dataframe')
    ml.randomizeRows(seed=0)

    if(False):
        print(ml.df.head(3))
        print('Missing values in columns:\n', ml.missingValues())
    ml.examine()    

    # TODO: Encode kaup.osa into features

    Vh_max = np.max(ml.df['Vh'])
    print('max Vh:', Vh_max, 'euro')

    print('Correlation matrix:')
    print(ml.df.corr())

    print("---------------------------------------------")
    print("\n*Study standard ML models")
    if(False):
        study_standard_ml(ml)

    print("---------------------------------------------")
    print("\n*Study neural network for regression")
    if(True):
        study_nn(ml)

    print_results_for_reference()


    print('------------\nDone')




if __name__ == '__main__':
    main()