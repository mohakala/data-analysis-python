# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:43:31 2018

Study different optimization tasks following
  https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

@author: mhaa
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


"""
A. Unconstrained optimization of multivariate scalar 
functions (minmization)
"""

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def ownfunc2d(x):
    y = x[0]**2 + -10*x[0]*np.cos(x[0]) - x[1]**3 + 9*x[1]*np.sin(x[1]) + 50 
    return(y)


def nelder_mead():
    # Without gradient
    #
    # Nelder-Mead Simplex algorithm. 
    # Probably the simplest way to minimize a 
    # fairly well-behaved function. Requires only 
    # function evaluations, good choice for simple 
    # minimization problems. No gradients, may take 
    # longer to find the minimum.
    #
    # Another option: method='powell'
    
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    print('x0:', x0)
    res = minimize(rosen, x0, method='nelder-mead',
            options={'xtol': 1e-8, 'disp': True})
    print(res.x)

    x0 = np.array([0.0, 0.0])
    res = minimize(ownfunc2d, x0, method='nelder-mead',
            options={'xtol': 1e-8, 'disp': True})
    print(res.x)


def BFGS():
    # Needs gradient information
    #   either directly or estimated numerically from the function 
    print('BFGS')
    
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'disp': True})
    print('With gradient information:', res.x)

    res = minimize(rosen, x0, method='BFGS', 
               options={'disp': True})
    print('Gradient estimated numerically:', res.x)


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
                -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp


def Newton_CG():
    # Suitable for large-scale problems
    # 
    print('Newton-CG')
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='Newton-CG',
                   jac=rosen_der, hess=rosen_hess,
                   options={'xtol': 1e-8, 'disp': True})
    print(res.x)
    
    # Also version with Hessian product
    #   need Hp
    # For larger minimization problems, storing the entire 
    # Hessian matrix can consume considerable time and memory. 
    # The Newton-CG algorithm only needs the product of the
    # Hessian times an arbitrary vector.
    res = minimize(rosen, x0, method='Newton-CG',
                jac=rosen_der, hessp=rosen_hess_p,
                options={'xtol': 1e-8, 'disp': True})
    print(res.x)



def trust_region_version():
    # Methods suitable for large-scale problems
    # (problems with thousands of variables)
    #
    # Similar to the trust-ncg method, the trust-krylov method is 
    # a method suitable for large-scale problems as it uses 
    # the hessian only as linear operator by means of 
    # matrix-vector products. It solves the quadratic subproblem 
    # more accurately than the trust-ncg method.
    #
    print('Trust-region methods')
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    
    print('trust-ncg')
    res = minimize(rosen, x0, method='trust-ncg',
                   jac=rosen_der, hess=rosen_hess,
                   options={'gtol': 1e-8, 'disp': True})
    print(res.x)

    print('trust-krylov, trust-exact don''t work, maybe need scipy update')
    #res = minimize(rosen, x0, method='trust-krylov',
    #               jac=rosen_der, hess=rosen_hess,
    #               options={'gtol': 1e-8, 'disp': True})
    #print(res.x)


    # Trust-exact
    # For medium-size problems, where storage and factorization 
    # of Hessian are not critical. Also exact solution of trust-region subproblems
    #
    #res = minimize(rosen, x0, method='trust-exact',
    #               jac=rosen_der, hess=rosen_hess,
    #               options={'gtol': 1e-8, 'disp': True})
    #print(res.x)


"""
B. Constrained optimization of multivariate scalar 
functions (minmization)
"""


"""
C. Least-squares minimization (least_squares)
"""


"""
D. Univariate function minimizers (minimize_scalar)
     Unconstrained minimization (method='brent')
     Bounded minimization (method='bounded')
"""

def ownfunc(x):
    y = x**2 + 10*x*np.cos(x) + 2.5
    return(y)


def bessel_min():
    from scipy.special import j1

    #func = j1
    func = ownfunc
    
    x = np.linspace(-10, 15, 1000)
    plt.plot(x, func(x))
    plt.show()

    # Bounded minimization
    bounds = (-10, 0)
    res = minimize_scalar(func, bounds=bounds, method='bounded')
    print('Minimum at:', res.x, 'for bounds:', bounds, ', y =', func(res.x))

    # Unbounded minimization
    bracket = (-10, 0)  # Note bracket=(-10, 0) --> wrong minimimun
    #bracket = None
    res = minimize_scalar(func, bracket=bracket, method='brent')
    print('Minimum at:', res.x, 'unbounded, y =', func(res.x))
    
    # Note: Own func w/ unbounded optimization leads to wrong minimum
    # --> Use bounds if known
    
    

"""
E. Custom minimizers
"""


"""
F. Root finding
"""


"""
G. Basin hopping
"""




def main():
    
    nelder_mead()
    print('---------')    
    bessel_min()    
    print('---------')    
    BFGS()    
    print('---------')    
    trust_region_version()

if __name__ == '__main__':
    main()



