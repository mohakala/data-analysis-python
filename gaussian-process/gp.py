import numpy as np
from matplotlib import pyplot as plt


"""
# Study Gaussian Process Regression
"""


def plotGPRresults(gp, X, y, xtest=0, ytest_true=0, dy=0):
    """
    Plot results for the gp model (one-dimensional data X)
    dy = error in the training data
    X, y, dy = training data
    
    xtest = test data x-points
    ytest_true = true function at xtest-points
    """
    
    # Calculate the predictions
    y_pred, sigma = gp.predict(xtest, return_std=True)


    plt.figure()
        
    # Plot the predictions    
    plt.plot(xtest, y_pred, 'b-', label='predictions')
    plt.plot(xtest, y_pred - 2*sigma, 'b:')
    plt.plot(xtest, y_pred + 2*sigma, 'b:')


    # Plot the training data, with errorbar dy if it's available
    if(np.sum(dy) != 0):       
        plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'observations')
    else:
        plt.plot(X, y, 'r.', markersize=10, label='training data')


    # Plot the true function if available
    if(np.sum(ytest_true) != 0):  
        plt.plot(xtest, ytest_true, 'r:', label='true function')

    
    plt.xlabel('coordinate')
    plt.ylabel('energy')
    plt.ylim(np.min(y)*0.9, np.max(y)*1.1)
    plt.legend()
    plt.show()



def function(X):
    return (np.sin(1.5*X)*0.5*X).ravel()



# A. Datapoints, no noise

# Training data A
X = np.atleast_2d([1, 5, 6, 7, 8, 10]).T 
y = function(X).ravel()

# Training data B
X1 = np.linspace(-4, 5, 10).reshape(-1,1)
y1 = np.array([1, 1.8, 3.2, 3.9, 4.5, 5.1, 3.8, 3.6, 3.5, 2.1]).ravel()

# Choose which training data to use
X = X1
y = y1

print('Xtrain', X)
print('ytrain', y)                 
 
                 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel





# Version 1 with target's known noise level
kernel = C(1.0) * RBF(10, (1e-1, 1e2))
alpha = 0.1
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)


# Print kernel information
if(False):
    print('Kernel information:')
    testC = C(1.0)
    testRBF = RBF(10, (1e-1, 1e2))
    print(testC.get_params())
    print(testRBF.get_params())


# Fit the hyperparameters
gp.fit(X, y)
print('\nOptimized kernel:', gp.kernel_)
print('Log marginal likelihood:', gp.log_marginal_likelihood_value_)
    
    

# Version 2 with White Kernel
kernel = C(1.0) * RBF(1, (1e-1, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)

# Fit the hyperparameters
gp.fit(X, y)
print('\nOptimized kernel:', gp.kernel_)
print('Log marginal likelihood:', gp.log_marginal_likelihood_value_)





# Plot predictions, data A 
xtest = np.atleast_2d(np.linspace(0, 10, 100)).T
ytest_true = function(xtest)

# Plot predictions, data B
xtest = np.atleast_2d(np.linspace(-5, 5, 50)).T


#plotGPRresults(gp, X, y, xtest, ytest_true)
plotGPRresults(gp, X, y, xtest)



# B. Datapoints with noise

# Add noise to y
dy = 1.0 + 0.0*y
noise = np.random.normal(0, dy)
y = y + noise

kernel = C(1.0) * RBF(10, (0.5, 1e2))  ## 0.5? study more

# To study: result as a function of regularization parameter (noise level)  alpha
alpha = 0.2
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
gp.fit(X, y)
print('\nOptimized kernel:', gp.kernel_)
print('Alpha:', gp.alpha_)


# Plot predictions
xtest = np.atleast_2d(np.linspace(0, 10, 100)).T
ytest_true = function(xtest)
plotGPRresults(gp, X, y, xtest, ytest_true, dy)


