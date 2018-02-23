# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:54:03 2018

Matrix decompositions following
  https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/

@author: mhaa
"""

from numpy import array
import numpy as np


def LU_decomp():
    # LU decomposition
    # L: lower triangular matrix
    # U: upper triangular matrix
    # For linear equations to simplify
    # To find determinants and inverse of a matrix
    from scipy.linalg import lu
    # define a square matrix
    A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print('A:', A)
    # LU decomposition
    P, L, U = lu(A)
    print('P, L, U')
    print(P)
    print(L)
    print(U)
    # reconstruct
    print('Reconstructed')
    B = P.dot(L).dot(U)
    print(B)


def QR_decomp():
    # QR decomposition
    # For n x m matrices
    # Q: m x m matrix
    # R: upper triangle m x n
    # Used to solve systems of linear equations
    from numpy import array
    from numpy.linalg import qr
    # define a 3x2 matrix
    A = array([[1, 2], [3, 4], [5, 6]])
    print(A)
    # QR decomposition
    Q, R = qr(A, 'complete')
    print(Q)
    print(R)
    # reconstruct
    B = Q.dot(R)
    print(B)


def Cholesky():
    # Cholesky decomposition
    # For square symmetric matrices, all elements > 0
    # (positive definite)
    # A = LL^T or A = U^T . U
    # Used for simulation and optimization methods
    # For symmetric matrices nearly twice as efficient as QR
    from numpy import array
    from numpy.linalg import cholesky
    # define a 3x3 matrix
    A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    print(A)
    # Cholesky decomposition
    L = cholesky(A)
    print(L)
    # reconstruct
    B = L.dot(L.T)
    print(B)
    
    
def SVD_decomp():
    # A = U S V^H, V = vh, U = u
    # The 1D array s contains the singular values of a and u and vh are unitary
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)
    print(a.shape)

    # Reconstruction based on full SVD, 2D case:
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    print(u.shape, s.shape, vh.shape)
    np.allclose(a, np.dot(u[:, :6] * s, vh))
    smat = np.zeros((9, 6), dtype=complex)
    smat[:6, :6] = np.diag(s)
    np.allclose(a, np.dot(u, np.dot(smat, vh)))
    
    
def NMF_decomp():
    # Find two non-negative matrices (W, H) whose product 
    # approximates the non- negative matrix X. 
    #
    # This factorization can be used for example for 
    # dimensionality reduction, source separation or 
    # topic extraction.
    #
    # For example:
    #   https://www.slideshare.net/koorukuroo/nmf-with-python
    import numpy as np
    X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    print('X:', X)
    from sklearn.decomposition import NMF
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(X)
    print('W:', W)
    H = model.components_
    print('H:', H)
    B = W.dot(H)
    print('Reconstructed:', B)


    
def main():
    LU_decomp()
    print('===========')
    QR_decomp()
    print('===========')
    Cholesky()
    print('===========')
    SVD_decomp()
    print('===========')
    NMF_decomp()
    
    
    




if __name__ == '__main__':
    main()
