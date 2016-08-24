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
    # for an array x returns array y:
    # th_i=1 if x_i > 0, otherwise th_i=0 
    y=(np.sign(x)+1)/2
    # must repair the value y=0.5 corresponding to x=0
    y[y==0.5] = 0
    return(y)

def test_heaviside(): # (call from main)
    test=np.array([-1,-0.01,0,0.02,1])
    print('test heaviside:',test)
    print(heaviside(test))        

def sigmoid(x):
    beta = 1.0
    y=1/(1 + np.exp(-beta * x))
    return(x)


def runPerceptron(inp,w):
    # inp: column vector, length: m
    # w: weight matrix m x n for n neurons.
    # Each neuron gets m input values.
    # returns: values (0/1) of the n neurons 
    # print('runPerceptron')
    # matrix product to get neurons' outputs:
    # y_n = w_nm x inp_m
    out=np.mat(w.T) * np.mat(inp)
    #print('Aux output:',out)
    out=heaviside(out)
    return(out)

def fwdPhase(inp,w,acttype):
    # Same as runPerceptron, except with 'acttype' added
    # inp: column vector, length: m
    # w: weight matrix m x n for n neurons.
    # Each neuron gets m input values.
    # returns: values (0/1) of the n neurons 
    # print('runPerceptron')
    # matrix product to get neurons' outputs:
    # y_n = w_nm x inp_m
    out=np.mat(w.T) * np.mat(inp)
    #print('Aux output:',out)
    if (acttype=='step'):
        out=heaviside(out)
    elif (acttype=='sigmoid'):
        out=sigmoid(out)
    elif (acttype=='linear'):
        # linear activation, no transformation
        pass
    else:
        sys.exit("Wrong acttype")
    return(out)


    

def trainPerceptronOneRound(w,inp,target,eta):
    # One round for all input datavectors
    nInputs=len(inp[0]) # number of inputs
    nInputDim=len(inp) # dimension of input vector
    nNeur=len(w[0]) # number of neurons

    changeForInputSet=0

    for i_inp in range(nInputs): # Loop over input data
        oneInput=inp[:,i_inp].reshape((3,1)) # TODO: REPAIR 3,1?
        # print('one input:',i_inp,oneInput)
        out=runPerceptron(oneInput,w)
        for i_neur in range(nNeur): # Loops of neurons
            for i in range(nInputDim): # Loop over data dimensions
                error=(target[i_neur,i_inp]-out[i_neur])
                change=eta*( error )*oneInput[i]
                w[i,i_neur]=w[i,i_neur]+change # update weights of neurons
                changeForInputSet+=np.absolute(change)
    print('Abs. change for input set =',changeForInputSet)
    return(w,changeForInputSet)


def trainMLPOneRound(v,w,inp,target,eta):

    # Forwards phase
    hid=fwdPhase(inp[:,0].reshape((3,1)),v,'sigmoid')
    print('Output at the hidden layer:\n',hid)
    # must add the -1 bias to use hid as input to output layer
    inp2=np.array([[-1],hid[0],hid[1]])
    out=fwdPhase(inp2,w,'sigmoid')
    print('Output at the output layer:\n',out)

    # Backards phase
    # TODO
    # errOut=() ... # do the loops
        
    # TODO
    # TODO
    changeForInputSet=0
    return(v,w,changeForInputSet)


def trainMLP(v,w,inp,target,eta):
    i=-1
    while True:
        i+=1
        print('Iteration:',i)
        v,w,changeBetweenIters = trainMLPOneRound(v,w,inp,target,eta)
        if (changeBetweenIters==0):
            break
    return(v,w)

def trainPerceptron(w,inp,target,eta):
    i=-1
    while True:
        i+=1
        print('Iteration:',i)
        w,changeBetweenIters = trainPerceptronOneRound(w,inp,target,eta)
        if (changeBetweenIters==0):
            break
    return(w)



if __name__ == '__main__':

    # A. Perceptron section

    # Parameters and inputs
    inp=np.array([[-1,0,0],[-1,0,1],[-1,1,0],[-1,1,1]]).T
    # inp=np.array([[1,0],[2,0],[1,2]]).T  # test, 2x3 matrix
    target=np.array([[0,1,1,1],[0,1,1,1]]) # one row for each neuron
    eta=0.25 # learning rate
    # w=(np.random.rand(3,1)-0.5)*0.1 # initial random weights
    # w=np.array([[-0.05,-0.02,0.02]]).T
    # Book: w=np.array([[-0.05,-0.02,0.02],[-0.05,-0.02,0.02]]).T

    # Output layer weights
    w=(np.random.rand(3,2)-0.5)*0.1  

    # Hidden layer weights
    v=(np.random.rand(3,2)-0.5)*0.1  



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
    print('Output layer final weights:\n',w)

    # Check perceptron for the input data
    print('-- Check that the perceptron works:')
    for i in range(4):
        print('Input:\n',inp[:,i])
        print('Output:\n',fwdPhase(inp[:,i].reshape((3,1)),w,'step'))

    # Another test
    testinput=np.array([-1,0.3,0.3]).reshape(3,1)
    print('Another test, testinput:\n',testinput,'\n Output:\n',runPerceptron(testinput,w))



    # B. MLP section

    # Parameters and inputs
    inp=np.array([[-1,0,0],[-1,0,1],[-1,1,0],[-1,1,1]]).T
    target=np.array([[0,1,1,1],[0,1,1,1]]) # one row for each neuron
    eta=0.25 # learning rate

    # Hidden layer weights
    v=(np.random.rand(3,2)-0.5)*0.1  
    # Output layer weights
    w=(np.random.rand(3,2)-0.5)*0.1  

    # Train MLP
    print('-- Train MLP')
    trainMLP(v,w,inp,target,eta)
    print('Hidden layer v:\n',v)
    print('Output layer w:\n',w)

    # Running a trained MLP
    out=fwdPhase(inp[:,0].reshape((3,1)),v,'sigmoid')
    print('Output at the hidden layer:\n',out)
    # must add the -1 bias to use out as inp2
    inp2=np.array([[-1],out[0],out[1]])
    out2=fwdPhase(inp2,w,'sigmoid')
    print('Output at the output layer:\n',out2)








#   C. dump, just tests
    aaatrue=np.array([0,1,0,1])
    bbbpred=np.array([1,1,1,0])

    instance=analyzeBinaryPreds(aaatrue,bbbpred)   # print(type(instance))
    instance.calculateResults()
    # instance.showResults()






