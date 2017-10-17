# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:55:20 2017

@author: mhaa
"""

import numpy as np
import matplotlib.pyplot as plt


class markov(object):
    """
    Items for Markov chains
    """
    
    def __init__(self, tm=None):
        """
        Use input transition matrix
        """
        if tm is not None:
            self.tm = tm
            self.check_tm()


    def check_tm(self):
        """
        Check that the matrix is a square matrix
        Check that all the rows of transition matrix sum to 1.0
        """
        assert self.tm.shape[0]==self.tm.shape[1], "not a square matrix"
        l = len(self.tm)
        for i in range(l):
            assert np.sum(self.tm[i, :])==1.0, "row %i in transition matrix does not sum to 1.0" % i
            
            
    def make_trans(self, state_init, n=1):
        # make n transitions:
        # state = state x TM x TM ...
        state = state_init        
        for i in range(n):
            state = np.matmul(state, self.tm)
            # alternatives:
            # state = state.dot(self.tm)
        return(state)
    
        
    def converged_state(self, state):
        """
        Iterate until changes in state's norm < epsilon
        """
        epsilon=0.0000001

        norm1 = np.linalg.norm(state)
        i=0
        while i < 100:
            i=i+1
            state = self.make_trans(state)
            norm2 = np.linalg.norm(state)
            # print('i, state, norm1, norm2:', i, state, norm1, norm2)
            if(np.abs(norm2-norm1)/norm1 < epsilon):
                print('>convergence in ', i, 'steps')
                return(state)
            norm1 = norm2
        print('i=100 reached, no convergence in norm')


    def matrixprop(self, matrix):
        v, w = np.linalg.eig(matrix)
        print('v:', v)
        print('w:', w)
        det = np.linalg.det(matrix)
        print('determinant:', det)


def normal_markovchain():
    # Transition matrix
    # [A to A, A to B]
    # [B to A, B to B]

    # Example 1
    # https://www.analyticsvidhya.com/blog/2014/07/markov-chain-simplified/
    tm = np.array([
            [0.7, 0.3],
            [0.1, 0.9]
            ])

    state = np.array([0.55, 0.45])

    
    # Example 2
    # https://www.analyticsvidhya.com/blog/2014/07/solve-business-case-simple-markov-chain
    tm = np.array([
            [0.7, 0.05, 0.03, 0.22],
            [0.05, 0.55, 0.35, 0.05],
            [0.0, 0.00, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
            ])
    
    state = np.array([0.60, 0.40, 0.0, 0.0])

    # Example 2b
    state = np.array([1.0, 0.0, 0.0, 0.0])

    # Example 2c
    state = np.array([0.0, 1.0, 0.0, 0.0])

    # Example 3: rain, no rain
    tm = np.array([
            [0.5, 0.5],
            [0.05, 0.95]
            ])
    state = np.array([0.0, 1.0])

    mk = markov(tm)
    
    # Old and new state 
    print('old state:', state)
    print('new state:', mk.make_trans(state, 1))

    # Converged state
    print('converged state:', mk.converged_state(state))
    
    # Study the transition matrix
    # mk.matrixprop(tm)

 
def hidden_markovchain():
#    from hmmlearn import hmm
    from hmmlearn.hmm import GaussianHMM
    
    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
    # 2-dimensional
    means2 = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
    
    covars2 = .5 * np.tile(np.identity(2), (4, 1, 1))

    # 1-dimensional
    means=np.array([[0.0],
                  [5.0],
                  [50.0],
                  [100.0]])
    covars = .5 * np.tile(np.identity(1), (4, 1, 1))

    

#    model = hmm.GaussianHMM(n_components=4, covariance_type="full")
    model = GaussianHMM(n_components=4, covariance_type="full")

    # Set the parameters
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # Generate samples
    X, Z = model.sample(15)
    print(X, Z)



    if(False):    
        plt.figure()
        plt.plot(X)
        plt.plot(Z)
        plt.show()


    # Estimate optimal sequence of hidden states
    remodel = GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
    remodel.fit(X)  
    print(remodel.score)
    Z2 = remodel.predict(X)
    print(Z2)
    print(remodel.monitor_)
    print('Converged:', remodel.monitor_.converged)
    
    print("Predictions:")
    for i in range(remodel.n_components):
        print("mean = ", remodel.means_[i])
        print("var = ", np.diag(remodel.covars_[i]))

    
    


def main():    
    if(False):
        normal_markovchain()
        
    if(True):
        hidden_markovchain()
    
    print('\nDone')    

    
if __name__ == '__main__':
    main()



