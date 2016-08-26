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
    beta = 0.05
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
                # TODO: take this outside the i-loop
                error=(target[i_neur,i_inp]-out[i_neur])
                change=eta*( error )*oneInput[i]
                w[i,i_neur]=w[i,i_neur]+change # update weights of neurons
                changeForInputSet+=np.absolute(change)
    print('Abs. change for input set =',changeForInputSet)
    return(w,changeForInputSet)


def trainMLPOneRound(v,w,inp,target,eta):
    nInputs=len(inp[0]) # number of inputs
    nInputDim=len(inp) # dimension of input vector
    nNeurW=len(w[0]) # number of neurons
    nNeurV=len(v[0]) # number of neurons

    changeForInputSet=0

    for i_inp in range(nInputs): # Loop over input data
        oneInput=inp[:,i_inp].reshape((nInputDim,1)) 

        # Forward phase
    
        hid=fwdPhase(oneInput,v,'sigmoid') # hidden layer
        #print('Output at the hidden layer:\n',hid)
        # must add the -1 bias to use hid as input to output layer
        hidWithBias=np.array([[-1],hid[0],hid[1]])
#        hidWithBias=np.array([[-1],hid[0]]) # For one neuron
        #print(hidWithBias,'\n',w)
        out=fwdPhase(hidWithBias,w,'sigmoid') # output layer
        #print('Output at the output layer:\n',out)

        # Backward phase

        errOut=np.zeros((nNeurW,1))
        errHid=np.zeros((nNeurV,1))

        # Error at output layer
        for i_neurW in range(nNeurW): # Loop over output neurons
            #print('i_neur:',i_neurW)
            errOut[i_neurW]=(target[i_neurW,i_inp]-out[i_neurW]) * out[i_neurW] * (1-out[i_neurW])
        #print('errOut:',errOut)

        # Error at hidden layer
        for i_neurV in range(nNeurV): # Loop over hidden neurons
            auxiliarySum=0.0
            # TODO: Here the sums can be in a more clever way,
            # TODO: some of the avoided
            for j_neurW in range(nNeurW):
                # notice: w[i_neurV+1,j_neurW], because bias is not
                # and must not be included
                auxiliarySum+=w[i_neurV+1,j_neurW]*errOut[j_neurW]
            errHid[i_neurV]=hid[i_neurV]*(1-hid[i_neurV])*auxiliarySum

        # Update output layer weights
        for i_neurW in range(nNeurW): # Loop over output neurons
            for i in range(nInputDim): # Loop over data dimensions
                change=eta*errOut[i_neurW]*hidWithBias[i]
                w[i,i_neurW]=w[i,i_neurW]-change
                changeForInputSet+=np.absolute(change)

        # Update hidden layer weights
        for i_neurV in range(nNeurV): # Loop over hidden neurons
            for i in range(nInputDim): # Loop over data dimensions
                change=eta*errHid[i_neurV]*oneInput[i]
                w[i,i_neurV]=w[i,i_neurV]-change
                changeForInputSet+=np.absolute(change)
             
            
        # TODO SHUFFLE INPUTS, CLEAN
    
    return(v,w,changeForInputSet)


def trainMLP(v,w,inp,target,eta):
    i=0
    while True:
        i+=1
        v,w,changeBetweenIters = trainMLPOneRound(v,w,inp,target,eta)
        print('Iteration:',i,'Change:',changeBetweenIters)
        if ( (changeBetweenIters<1e-4) or (i>1000) ):
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


def perceptronSection():
    # MORE LIKE TESTS, RAW VERSION
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



if __name__ == '__main__':

    # A. Perceptron section
    # perceptronSection()


    # B. MLP section

    # TODO: To not need to include -1's in the input
    # TODO: Also one neuron or more than two neurons
    # TODO: Shuffle
    
    nNeuronsW = 2 # number of neurons, output layer
    nNeuronsV = 2 # number of neurons, hidden layer
    
    # Parameters and inputs
    eta=0.25 # learning rate
    inp=np.array([[-1,0,0],[-1,0,1],[-1,1,0],[-1,1,1]]).T  

    nInputs=len(inp[0]) # number of inputs
    nInputDim=len(inp) # dimension of input vector

#    target=np.array([[0,1,1,1],[0,1,1,1]]) # OR # one row for each neuron
    target=np.array([[0,1,1,0],[0,1,1,0]]) # XOR
#    target=np.array([[0,1,1,1]]) # OR
#    target=np.array([[0,1,1,0]]) # XOR


    # Hidden and output layer weights
    v=(np.random.rand(nInputDim,nNeuronsV)-0.5)*0.1  
    w=(np.random.rand(nInputDim,nNeuronsW)-0.5)*0.1  

    # Train MLP
    print('-- Train MLP')
    trainMLP(v,w,inp,target,eta)
    print('Results:\nHidden layer v:\n',v)
    print('Output layer w:\n',w)

    # Running a trained MLP over the input vectors
    for i_inp in range(nInputs):
        out=fwdPhase(inp[:,i_inp].reshape((nInputDim,1)),v,'sigmoid')
        # print('Output at the hidden layer:\n',out)
        # must add the -1 bias to use out as inp2
        inp2=np.array([[-1],out[0],out[1]])
        out2=fwdPhase(inp2,w,'sigmoid')
        print('Inp, Output at the output layer:',i_inp,'\n',out2)








#   C. dump, just tests
    aaatrue=np.array([0,1,0,1])
    bbbpred=np.array([1,1,1,0])

    instance=analyzeBinaryPreds(aaatrue,bbbpred)   # print(type(instance))
    instance.calculateResults()
    # instance.showResults()






