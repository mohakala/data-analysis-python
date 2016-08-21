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


# TODO: More than one neuron

# m inputs
# n neurons

def heaviside(x):
    # for an array returns array th:
    # th_i=1 if x_i > 0, otherwise th_i=0 
    th=(np.sign(x)+1)/2
    # must repair the value th=0.5 corresponding to x=0
    th[th==0.5] = 0
    return(th)

def test_heaviside(): # (call from main)
    test=np.array([-1,-0.01,0,0.02,1])
    print('test heaviside:',test)
    print(heaviside(test))        


def runPerceptron(inp,w):
    # inp: column vector, length: m
    # w: weight matrix m x n for n neurons.
    # Each neuron gets m input values.
    # returns: values (0/1) of the n neurons 0/1 
    # print('runPerceptron')
    # matrix product to get neurons' outputs:
    # y_n = w_nm x inp_m
    out=np.mat(w.T) * np.mat(inp)
    #print('Aux output:',out)
    return(heaviside(out))
    

def trainPerceptronOneRound(w,inp,target,eta):
    # One round for all input datavectors
    nInputs=len(inp[0]) # number of inputs
    nInputDim=len(inp) # dimension of input vector
    nNeur=len(w[0]) # number of neurons

    changeForInputSet=0
    for i_inp in range(nInputs):
        #print('Input datavector:',i_inp)
        oneInput=inp[:,i_inp].reshape((3,1))
        #print('one input:',oneInput)
        out=runPerceptron(oneInput,w)
        for i_neur in range(nNeur):
            for i in range(nInputDim): # update weights of neurons
                #print('Updating neuron:',i)
                change=eta*(target[i_neur,i_inp]-out[i_neur])*oneInput[i]
                w[i,i_neur]=w[i,i_neur]+change
                changeForInputSet+=np.absolute(change)
    print('Abs. change for input set =',changeForInputSet)
    return(w,changeForInputSet)


def trainPerceptron(w,inp,target,eta):
    i=-1
    while True:
        i+=1
        print('Iteration:',i)
        w,change = trainPerceptronOneRound(w,inp,target,eta)
        if (change==0):
            break
    return(w)


if __name__ == '__main__':
    # Parameters and inputs
    inp=np.array([[-1,0,0],[-1,0,1],[-1,1,0],[-1,1,1]]).T
    # inp=np.array([[1,0],[2,0],[1,2]]).T  # test, 2x3 matrix
    target=np.array([[0,1,1,1],[0,1,1,1]]) # one row for each neuron
    eta=0.25 # learning rate
    # w=(np.random.rand(3,1)-0.5)*0.1 # initial random weights
    # w=np.array([[-0.05,-0.02,0.02]]).T
    # Book: w=np.array([[-0.05,-0.02,0.02],[-0.05,-0.02,0.02]]).T
    w=(np.random.rand(3,2)-0.5)*0.1 # initial random weights

    #print('test P:\n',runPerceptron(inp[:,0].reshape((3,1)),w))

    # Print information
    print('-- System')
    print('Input:\n',inp,'\nTarget:',target,'Eta:',eta)
    m=len(inp[0]) # number of inputs (rows of w)
    n=len(w[0])   # neurons (columns of w)
    nInputDim=len(inp)
    print('Number of inputs:',m,'neurons:',n,'input dim:',nInputDim)
    print('Initial weights (input dim x neurons):\n',w)

    # Train perceptron
    print('-- Train perceptron')
    trainPerceptron(w,inp,target,eta)
    print('Final weights:\n',w)

    # Check perceptron for the input data
    print('-- Check that the perceptron works:')
    for i in range(4):
        print('Input:\n',inp[:,i])
        print('Output:',runPerceptron(inp[:,i].reshape((3,1)),w))

    # Another test
    testinput=np.array([-1,0.3,0.3]).reshape(3,1)
    print('Another test, testinput:\n',testinput,'\n Output:\n',runPerceptron(testinput,w))


#   dump, just tests
    aaatrue=np.array([0,1,0,1])
    bbbpred=np.array([1,1,1,0])

    instance=analyzeBinaryPreds(aaatrue,bbbpred)   # print(type(instance))
    instance.calculateResults()
    # instance.showResults()






