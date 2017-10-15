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
        Check that all the rows of transition matrix sum to 1.0
        """
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
                return(state)
            norm1 = norm2
        print('i=100 reached, no convergence in norm')


    def matrixprop(self, matrix):
        v, w = np.linalg.eig(matrix)
        print('v:', v)
        print('w:', w)
        det = np.linalg.det(matrix)
        print('determinant:', det)


def main():
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
    

    mk = markov(tm)
    
    # Old and new state 
    print('old state:', state)
    print('new state:', mk.make_trans(state, 1))

    # Converged state
    print('converged state:', mk.converged_state(state))
    
    # Study the transition matrix
    # mk.matrixprop(tm)


 
    
    print('\nDone')    
    
    
if __name__ == '__main__':
    main()



