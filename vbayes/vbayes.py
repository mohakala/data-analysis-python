# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:35:54 2017

From https://arxiv.org/pdf/1410.0870.pdf
http://bayespy.org/examples/gmm.html
"""

import os
cwd = os.getcwd()
print(cwd)

import  numpy as np
import  bayespy.plot as bpplt
from  bayespy  import  nodes

import pandas as pd


def load_data_xy():    
    rawdata='C:\\Python34\\datasets\\matkaaika_V1.csv'
    df = pd.read_csv(rawdata)

    # Delete unnecessary columns
    col_to_delete=['0', 'lunta', 'huono keli']
    df=df.drop(col_to_delete, 1)

    # Delete rows with missing values
    df = df.dropna()
    print(df.describe())
        
    X=df['min8'].values
    X=X.reshape(-1, 1)
    y=df['kesto'].values
    return(X, y)


def traveltime():
    X, y = load_data_xy()
    
    # Center the observations y to zero
    ymean = np.mean(y)
    y = y - ymean
    print('ymean = ', ymean, 'Centering y to zero')
    
    
    # Prior
    from bayespy.nodes import GaussianARD
    from bayespy.nodes import Gamma

    mu1 = GaussianARD(0, 0.05)
    tau1 = Gamma(1e-3, 1e-3)


    # Likelihood    
    # Data samples assumed from normal distribution
    y1 = GaussianARD(mu1, tau1, plates=(len(y),))
    y1.observe(y)


    print('Before observations:')
    print('get_moments, mu1:', mu1.get_moments())
    print('get_moments, tau1:', tau1.get_moments())


    # Update the posteriors directly
    mu1.update()

    print('After observations:')
    print('get_moments, mu1:', mu1.get_moments())
    print('get_moments, tau1:', tau1.get_moments())
    sigma = np.sqrt(mu1.get_moments()[1])
    print('Result: ymean +_ 2sigma =', ymean, '+_', 2*sigma)


    # Examine the posterior approximation for means of gaussians (mu)
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.pdf(mu1, np.linspace(-10, 10, num=300))
    bpplt.pyplot.title('PDF of mu')    




def time_to_event():
    """
    How long it takes that something happens?
      Complicating factor: censoring
    The samples are drawn from log-normal distribution
    Normal distribution: defined by mean and variance
    
    Gaussian prior - Gaussian likelihood
    """

    # Data
    data1=np.log([60, 74, 37, 45, 75, 40, 50, 50, 146, 70, 50, 84, 60, 149, 50])
    data2=np.log([50, 130, 100, 130, 50, 140, 129, 76, 138, 69, 70, 144])


    # Prior
    from bayespy.nodes import GaussianARD
    from bayespy.nodes import Gamma

    mu1 = GaussianARD(0, 1e-3)
    tau1 = Gamma(1e-3, 1e-3)
    mu2 = GaussianARD(0, 1e-3)
    tau2 = Gamma(1e-3, 1e-3)

    # Alternative prior
    if(True):
        mu1 = GaussianARD(0, 1e-6)
        tau1 = Gamma(1e-6, 1e-6)
        mu2 = GaussianARD(0, 1e-6)
        tau2 = Gamma(1e-6, 1e-6)


    # Plot the prior
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    # bpplt.pdf(tau1, np.linspace(1e-6, 0.00004, num=300))
    bpplt.pdf(mu1, np.linspace(1e-6, 1000, num=300))
    bpplt.pyplot.title('PDF of mu1')    



    # Likelihood    
    # Data samples assumed from log-normal distribution
    y1 = GaussianARD(mu1, tau1, plates=(len(data1),))
    y2 = GaussianARD(mu2, tau2, plates=(len(data2),))

    y1.observe(data1)
    y2.observe(data2)

    print('Before observations:')
    print('get_moments, mu1:', mu1.get_moments())
    
    
    
    # Inference engine
    #  not needed since we have a conjugate prior and likelihood
    if(False):
        from bayespy.inference import VB
        Q1 = VB(y1, mu1, tau1)
        Q1.update(repeat=100)

        Q2 = VB(y2, mu2, tau2)
        Q2.update(repeat=100)
        print('get_moments, mu2:', mu2.get_moments())


    # Update the posteriors directly
    mu1.update()
    mu2.update()

    print('After observations:')
    print('get_moments, mu1:', mu1.get_moments())


    # Examine the posterior approximation for means of gaussians (mu)
    bpplt.pyplot.figure()
    bpplt.pdf(mu1, np.linspace(0, 10, num=300))
    bpplt.pyplot.title('PDF of mu')    
    bpplt.pdf(mu2, np.linspace(0, 10, num=300))
    bpplt.pyplot.title('PDF of mu')    


    # Difference in data
    print('Delta:', np.exp(mu1.get_moments()[0]) - np.exp(mu2.get_moments()[0]))

    
    
def poisson():
    """
    Poisson data
    Gamma prior - Poisson likelihood
    
    Poisson distribution: Natural distribution of the number of 
    events occurring randomly over a given amount of time.
    
    Draw samples y1, y2,... from Pois(theta), theta = kill rate = 
      kills per day. Experiment observes samples y1=0, y2=2 ... per day
    Prior of theta: expectation of kill rate, say 0.5. Then median
    of prior distribution is at 0.5. expectation of kill rate 95% to 
    be less than 2. 
    --> theta ~ Gamma(a, b), a and b from equations:
      Pr(theta < 0.5 | a, b) = 0.50
      Pr(theta < 2   | a, b) = 0.95 
    --> get a and b --> prior for theta
    
    With this prior and likelihood, the posterior theta 
    can be found analytically 
    --> Get theta distribution and the 95% probability interval
    0.140, 0.468
    
    """
    # Prior: kill rate theta
    theta = nodes.Gamma(1.11, 1.61)
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.pdf(theta, np.linspace(0.01, 3, num=300))
    bpplt.pyplot.title(' ')  
    print('theta moments:', theta.get_moments())

    # Likelihood
    #   Number of a:s killed per day
    y = nodes.Poisson(theta, plates=(13,))
    
    # Print 13 random samples from the likelihood 
    print('Random samples:')
    print(y.random())

    # Observed data    
    data=[1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y.observe(data)

    # Update theta
    theta.update()

    # Print 13 random samples from posterior distribution 
    print(y.random())

    bpplt.pyplot.figure()
    bpplt.pdf(theta, np.linspace(0.01, 3, num=300))
    bpplt.pyplot.title(' ')    




def defective():
    """ 
    Binomial data 
    From Bayesian Ideas and Data Analysis
    
    Beta prior - Binomial likelihood
    """
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Relevant distribution: Binomial distribution 
    #   there are x% defective parts in the production process

    # Prior belief: 
    #  propoortion of scrap is in the range 5% to 15%
    # This can be approximated by the following Beta distribution 
    theta = nodes.Beta([12.05, 116.06])
    # print(theta.pdf(0.1)) # Probability of theta being 0.1

    # Plot the prior distibution for theta
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.pdf(theta, np.linspace(0.01, 0.150, num=300))
    bpplt.pyplot.title(' ')    

    # Number of parts examined
    n_trials = 2430

    # The number of defective parts has binomial distribution
    y = nodes.Binomial(n_trials, theta)

    # Observed n=219 defective parts
    data=219
    y.observe(data)

    # Update the prior to get the posterior distribution
    theta.update()

    # Plot the theta posterior distribution
    bpplt.pyplot.figure()
    bpplt.pdf(theta, np.linspace(0.01, 0.150, num=300))
    bpplt.pyplot.title(' ')    

    if(False): 
        for i in range(1):
            data=500
            y.observe(data)
            theta.update()
        # Plot the new theta distribution
        bpplt.pyplot.figure()
        bpplt.pdf(theta, np.linspace(0.01, 0.150, num=300))
        bpplt.pyplot.title(' ')    
    
    
    
def example_from_article():
    """ 
    Example from the paper 
    
    Gaussian prior - Gaussian likelihood
    """

    # Model
    from bayespy.nodes import GaussianARD
    mu = GaussianARD(0, 1e-6)
    from bayespy.nodes import Gamma
    tau = Gamma(1e-6, 1e-6)

    # Likelihood
    y = GaussianARD(mu, tau, plates=(12,))
    
    data = np.array([4.5, 3.9, 6.3, 5.6, 4.9, 2.8, 7.4, 6.1, 4.8, 2.1])

    # Other data to test
    data_zinc = np.array([4.20, 4.36, 4.11, 3.96, 
                     5.63, 4.50, 5.64, 4.38, 
                     4.45, 3.67, 5.26, 4.66])

    
    y.observe(data_zinc)
    
    # Inference engine
    from bayespy.inference import VB
    Q = VB(y, mu, tau)
    print('get_parameters:', Q.get_parameters(y, mu, tau))
    print(Q.has_converged())

    Q.update(repeat=100)
    
    # print('mu:', Q[mu]) # does not work
    print('tau:', Q[tau])
    print('get_moments, mu:', mu.get_moments())
    #print('get_moments:', y.get_moments())
    print('get_moments, tau:', tau.get_moments())

    
    # Examine the posterior approximation
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.pdf(Q[mu], np.linspace(0, 10, num=300))
    bpplt.pyplot.title('PDF of mu')    
    
    bpplt.pyplot.figure()
    bpplt.pdf(Q[tau], np.linspace(1e-6, 7, num=300))
    bpplt.pyplot.title('PDF of tau')    
    
    
    

    

def zinc():
    """ 
    Zinc content data (normal data)
    From Bayesian Ideas and Data Analysis
    
    Gaussian prior - Gaussian likelihood
    """
    data = np.array([4.20, 4.36, 4.11, 3.96, 
                     5.63, 4.50, 5.64, 4.38, 
                     4.45, 3.67, 5.26, 4.66])
    
    ## Model
    X = np.ones(len(data)).reshape(-1, 1)
    print(X)

    # Prior mean
    from bayespy.nodes import GaussianARD
    B = GaussianARD(4.75, 0.0163, shape=(1,))
    print('B:', B)

    # BX
    from bayespy.nodes import SumMultiply
    F = SumMultiply('i,i', B, X)
    print('F:', F)

    # Prior distribution for tau = sigma^-2
    from bayespy.nodes import Gamma
    tau = Gamma(1e-3, 1e-3)
    print('tau:', tau)


    # Noisy observations
    Y = GaussianARD(F, tau)
    print('Y:', Y)


    ## Inference
    
    Y.observe(data)

    # PDF of Y (?)
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.pdf(Y, np.linspace(1e-6,8,12), color='k')
    #bpplt.pdf(Y, data, color='k')

    bpplt.pyplot.title('PDF of Y (?)')

    # VB inference engine by using stochastic nodes
    from bayespy.inference import VB
    Q = VB(Y, B, tau)
    print('get_moments, B:', B.get_moments())
    Q.update(repeat=1000)
    print('After engine:')
    print('B:', B)
    print('get_moments, B:', B.get_moments())
    print('tau:', tau)
    
    
    ## Show results
    
    # New inputs
    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)

    # Prediction
    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.plot(Fh, x=xh, scale=1) # Prediction with scale STD's
    bpplt.plot(data, x=np.arange(len(data)), color='r', marker='x', linestyle='None') # Samples

    # PDF of tau
    bpplt.pyplot.figure()
    bpplt.pdf(Q[tau], np.linspace(1e-6,8,100), color='k')
    bpplt.pyplot.title('PDF of tau')




def traveltime_min():
    X, y = load_data_xy()

    # Prior distribution for tau = sigma^-2
    from bayespy.nodes import Gamma
    tau = Gamma(1e-3, 1e-3)
    print('tau:', tau)
    
    


def linreg():
    """ 
    http://bayespy.org/examples/regression.html 
    
    Gaussian prior - Gaussian likelihood
    """

    ## Data    
    k = 2 # slope
    c = 5 # bias
    s = 2 # noise standard deviation

    x = np.arange(10)
    y = k*x + c + s*np.random.randn(10)


    ## Model
    
    X = np.vstack([x, np.ones(len(x))]).T
    
    # Prior mean
    from bayespy.nodes import GaussianARD
    B = GaussianARD(0, 1e-6, shape=(2,))
    print('B:', B)

    # BX
    from bayespy.nodes import SumMultiply
    F = SumMultiply('i,i', B, X)
    print('F:', F)

    # Prior distribution for tau = sigma^-2
    from bayespy.nodes import Gamma
    tau = Gamma(1e-3, 1e-3)
    print('tau:', tau)

    # Likelihood 
    #   Noisy observations
    Y = GaussianARD(F, tau)
    print('Y:', Y)


    ## Inference
    
    Y.observe(y)

    # VB inference engine by using stochastic nodes
    from bayespy.inference import VB
    Q = VB(Y, B, tau)
    Q.update(repeat=1000)
    print('B:', B)
    print('tau:', tau)
    
    
    ## Show results
    
    # Prediction
    
    # New inputs
    xh = np.linspace(-5, 15, 100)
    Xh = np.vstack([xh, np.ones(len(xh))]).T
    Fh = SumMultiply('i,i', B, Xh)

    import bayespy.plot as bpplt
    bpplt.pyplot.figure()
    bpplt.plot(Fh, x=xh, scale=2) # Prediction with 2 STD's
    bpplt.plot(y, x=x, color='r', marker='x', linestyle='None') # Samples
    bpplt.plot(k*xh+c, x=xh, color='r') # True function


    # Plot PDF of noice parameter tau
    bpplt.pyplot.figure()
    bpplt.pdf(Q[tau], np.linspace(1e-6,1,100), color='k')
    # Alternative: Just bpplt.pdf(tau, np.linspace...)
    bpplt.pyplot.axvline(s**(-2), color='r');
    bpplt.pyplot.title('PDF of tau')


    # Plot regression parameters and the true values
    bpplt.pyplot.figure();
    bpplt.contour(B, np.linspace(1,3,1000), np.linspace(1,9,1000),
                  n=10, colors='k');
    bpplt.plot(c, x=k, color='r', marker='x', linestyle='None',
               markersize=10, markeredgewidth=2)
    bpplt.pyplot.xlabel(r'$k$');
    bpplt.pyplot.ylabel(r'$c$');
    bpplt.pyplot.title('Regression parameters')



    

    
def two_gaussians():
    """ 
    Two 2D-Gaussians at (0,0) and (2,2) 
    
    Dirichlet prior
    """
    N = 500; D = 2
    data = np.random.randn(N, D)
    data [:200 ,:] += 2*np.ones(D)

    # Maximum number of clusters
    K = 5

    # Initialize the means mu of the unknown clusters
    mu = nodes.Gaussian(np.zeros(D), 0.01* np.identity(D), plates =(K,))
    print(mu)
    print("np.identity(D):", np.identity(D))

    # Initialize the precisions Lambda of the unknown clusters
    #   Wishart distribution for conjugate prior
    Lambda = nodes.Wishart(D, D*np.identity(D), plates =(K,))
    print(Lambda)
    
    # Cluster probabilities: Dirichlet prior
    alpha = nodes.Dirichlet (0.01* np.ones(K))
    
    # Cluster assignments 
    z = nodes.Categorical(alpha, plates =(N,))

    # Observations from Gaussian mixture distribution
    y = nodes.Mixture(z, nodes.Gaussian , mu , Lambda)

    # Observations (data)
    y.observe(data)

    # Variational Bayesian inference engine
    from  bayespy.inference  import  VB
    Q = VB(y, mu , z, Lambda , alpha)

    # Random initial assignments
    z.initialize_from_random ()

    Q.update(repeat =200)

    bpplt.gaussian_mixture_2d(y, alpha=alpha)
    bpplt.pyplot.show()

    # Print the parameters of the approximate posterior distributions
    print(alpha)



def main():

    if(False):
        traveltime()
    
    if(False):
        time_to_event()
        
    if(True):
        poisson()
        
    if(False):
        defective()
        
    if(False):
        example_from_article()
       
    if(False):
        zinc()

    if(False):
        linreg()

    if(False):
        two_gaussians()



    print("Done")
    
if __name__ == '__main__':
    main()


