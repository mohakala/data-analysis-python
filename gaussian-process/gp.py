import numpy as np
from matplotlib import pyplot as plt




"""
# Study Gaussian Process Regression
"""


def plotGPRresults(gp, X, y, xtest, ytest_true=0, dy=0):
    """
    Plot results for the gp model (one-dimensional data X)
    dy = error in the training data
    """
    y_pred, sigma = gp.predict(xtest, return_std=True)
    plt.figure()
    
    # Predictions    
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
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.ylim(-10, 20)
    plt.legend()
    plt.show()



# A. Dataponts without noise

# Training data
X = np.atleast_2d([1, 5, 6, 7, 8, 10]).T

def function(X):
    return (np.sin(1.5*X)*0.5*X).ravel()

y = function(X).ravel()


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Choose kernel
kernel = C(1.0) * RBF(10, (1e-1, 1e2))

print('Kernel information for checking:')
testC = C(1.0, (1e-3, 1e3))
testRBF = RBF(10, (1e-2, 1e2))
print(testC.get_params())
print(testRBF.get_params())


# Fit the hyperparameters
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)
print('\nOptimized kernel:', gp.kernel_)


# Plot predictions with a dense x-mesh
xtest = np.atleast_2d(np.linspace(0, 10, 100)).T
ytest_true = function(xtest)
plotGPRresults(gp, X, y, xtest, ytest_true)


# B. Datapoints with noise

# Add noise to y
dy = 1.0 + 0.0*y
noise = np.random.normal(0, dy)
y = y + noise

kernel = C(1.0) * RBF(10, (0.5, 1e2))  ## 0.5? study more

# To study: result as a function of regularization parameter alpha
alpha = 0.2
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
gp.fit(X, y)
print('\nOptimized kernel:', gp.kernel_)
print('Alpha:', gp.alpha_)


# Plot predictions with dense x-mesh
xtest = np.atleast_2d(np.linspace(0, 10, 100)).T
ytest_true = function(xtest)
plotGPRresults(gp, X, y, xtest, ytest_true, dy)


