# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:05:47 2017

@author: mhaa
"""

# Study https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
# Other references: 
#   http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
#   https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV



def study_linear_and_rbf(X, y):
    C=1
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    make_plot(X, y, svc)

    # C:     regularization term for soft margin cost function
    # gamma: small gamma: large influence of x_i on x_j
    
    gamma=1.0
    C=1
    svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X, y)
    print(svc)
    make_plot(X, y, svc)



def make_plot(X, y, svc):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)    
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC')
    plt.show()



def find_best_hyperparameters(X, y):
    # Cross validation
    C_range = np.logspace(-2, 10, 3)  # was: 13
    gamma_range = np.logspace(-9, 3, 3)
    print('C_range:', C_range)
    print('gamma_range:', gamma_range)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    # Study some examples by setting gamma and C by hand
    study_linear_and_rbf(X, y)
        
    # Find best gamma and C by cross validation
    find_best_hyperparameters(X, y)



    print('Done')    



if __name__ == '__main__':
    main()

    
