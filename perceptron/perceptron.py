import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
from mfunc import *  # Some own functions
sys.path.insert(0, 'C:\\Python34\\data-analysis-python\\random-forest-python')
from rf_titanic import *  # For confusion matrix

# print(sys.path)


# m inputs
# n neurons

def heaviside(x):
    # for an array returns array th:
    # th_i=1 if x_i > 0, otherwise th_i=0 
    th=(np.sign(x)+1)/2
    # must repair the value th=0.5 corresponding to x=0
    th[th==0.5] = 0
    return(th)

def test_heaviside(): # call from main
    test=np.array([-1,-0.01,0,0.02,1])
    print('test heaviside:',test)
    print(heaviside(test))        


def runPerceptron(inp,w):
    # inp: column vector, length: m
    # w: weight matrix m x n
    # returns: values of the n neurons 0/1 
    print('runPerceptron')
    m=len(w) # number of inputs (rows of w)
    n=len(w[0]) # neurons (columns of w)
    m2=len(inp) # inputs (alternative)
    print('Inputs:',m2,'Inputs from w:',m,'Neurons:',n)
    out=np.mat(w.T) * np.mat(inp) # matrix product to get neurons' outputs
    print('Aux output:',out)
    return(heaviside(out))
    

def trainPerceptron(w,inp,out,target,eta):

    # TODO Loop over inputs
    # TODO Convergence criterion
    
    out=runPerceptron(inp,w)

    for i in range(3): # update weights
        w[i]=w[i]+eta*(target-out)*inp[i]
    # print(w) 

    out=runPerceptron(inp,w)

    return(w)


if __name__ == '__main__':
    inp=np.array([[-1,0,0]]).T
    target=np.array([0])
    eta=0.25 # learning rate
    print('Input:',inp,'Target:',target,'Eta:',eta)
    
    # example=np.array([[1,0],[2,0],[1,2]]).T  # 2x3 matrix

    # Run perceptron once
    # w=(np.random.rand(3,1)-0.5)*0.1 # initial random weights
    w=np.array([[-0.05,-0.02,0.02]]).T
    out=runPerceptron(inp,w)
    print('Output:',out)

    trainPerceptron(w,inp,out,target,eta)




#   Just tests
    aaatrue=np.array([0,1,0,1])
    bbbpred=np.array([1,1,1,0])

    instance=analyzeBinaryPreds(aaatrue,bbbpred)   # print(type(instance))
    instance.calculateResults()
    # instance.showResults()






