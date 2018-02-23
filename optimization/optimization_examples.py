# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:43:31 2018

Optimization examples following
  https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

@author: mhaa
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt



def main():
    from scipy.special import j1

    x = np.linspace(0, 50, 1000)
    plt.plot(x, j1(x))
    plt.show()
    



    res = minimize_scalar(j1, bounds=(4, 7), method='bounded')
    print('Minimum at:', res.x)
    
    

if __name__ == '__main__':
    main()



