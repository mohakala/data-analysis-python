import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
from mfunc import *  # Some own functions

# m inputs
# n neurons

def runPerceptron(inp,w):
    m=len(w) # inputs
    n=len(w[0]) # neurons
    m2=len(inp) # inputs (alternative)
    print('Inputs:',m2,'Inputs from w:',m,'Neurons:',n)
    # return(out)
    

def trainPerceptron(inp,out,target):
    pass
    # set random values for w
    # 
    # return(w)


if __name__ == '__main__':
    x=np.array([1,0])

#    w=np.array([[1,0],[2,0],[1,2]]).T
    w=(np.random.rand(2,3)-+0.5)*0.1
    runPerceptron(x,w)





