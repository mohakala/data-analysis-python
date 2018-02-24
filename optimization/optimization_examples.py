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
from scipy.optimize import basinhopping

import matplotlib.pyplot as plt


"""
A. Unconstrained optimization of multivariate scalar 
functions (minimization)
"""

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rosen_der(x):
    # Gradient of the Rosen function 
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def ownfunc2d(x):
    # Own function with various local minima
    y = x[0]**2 + -10*x[0]*np.cos(x[0]) - x[1]**3 + 9*x[1]*np.sin(x[1]) + 50 
    return(y)


def nelder_mead():
    # Without gradient
    #
    # Nelder-Mead Simplex algorithm. 
    # Probably the simplest way to minimize a 
    # fairly well-behaved multivariate function. Requires only 
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
    # Takes gradient information
    # either directly or estimated numerically from the function 
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
functions (minimization)
"""

def func(x, sign=1.0):
    """ 
    Objective function 
    Use sign = -1 to make the task as maximization
    """
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)


def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])


def constr_opt():
    print('Constrained optimization')
    # Constraints
    # x^3 = y
    # y-1 >= 0
    # [x0, y0] = [-1.0, 1.0] initial guess
    # args=(-1.0) feed in the sign to get maximization problem
    # 
    # SLSQP - Sequential Least SQuares Programming optimization
    #
    cons = ({'type': 'eq',
             'fun' : lambda x: np.array([x[0]**3 - x[1]]),
             'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
         'jac' : lambda x: np.array([0.0, 1.0])})

    # First unconstrained optimization
    res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
                   method='SLSQP', options={'disp': True})
    print(res.x)
    
    # With constrains
    res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
                   constraints=cons, method='SLSQP', options={'disp': True})
    print(res.x)

    # With constrains, without gradient
    res = minimize(func, [-1.0,1.0], args=(-1.0,), 
                   constraints=cons, method='SLSQP', options={'disp': True})
    print(res.x)

    



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


def minim_scalar():
    # Optimization methods for univariate functions
    # Bounded and unbounded minimization

    # First Bessel function
    #from scipy.special import j1
    #func = j1

    # Own 1D function defined above to minimize
    func = ownfunc
    
    x = np.linspace(-10, 15, 1000)
    plt.plot(x, func(x))
    plt.show()

    # Bounded minimization
    bounds = (-10, 0)
    res = minimize_scalar(func, bounds=bounds, method='bounded')
    print('Minimum at:', res.x, 'for bounds:', bounds, ', y =', res.fun)

    # Unbounded minimization
    bracket = (-10, 0)  # Note bracket=(-10, 0) --> wrong minimimun
    #bracket = None
    res = minimize_scalar(func, bracket=bracket, method='brent')
    print('Minimum at:', res.x, 'unbounded, y =', res.fun)
    
    # Note: Own func w/ unbounded optimization leads to wrong minimum
    # --> Use bounds if known
    
    

"""
E. Custom minimizers
"""


"""
F. Root finding
"""


"""
G. Basin hopping: multivariate scalar functions

https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.basinhopping.html
"""

def print_fun(x, f, accepted):
    # Custom callback function, prints the value of every 
    # minimum found
    print("at x = %.3f minimum %.4f accepted %d" % (x, f, int(accepted)))



def basin_hopping_1d():
    # Stochastic global optimization. Particularly useful 
    # when the function has many minima separated by large 
    # barriers.
    #
    # No way to determine if 
    # the true global minimum has actually been found. Instead, 
    # as a consistency check, the algorithm can be run from a 
    # number of different random starting points to ensure the 
    # lowest minimum found in each example has converged to the 
    # global minimum. For this reason basinhopping will by 
    # default simply run for the number of iterations niter 
    # and return the lowest minimum found. 

    # 1D function
    print('1D function')
    func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    x0=[1.]
    print('Starting x0:', x0)
    
    x = np.linspace(-2, 2, 100)
    plt.plot(x, func(x))
    plt.show()
  
    
    minimizer_kwargs = {"method": "BFGS"}
    ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
                       niter=20, callback=print_fun)
    # niter: was 200 originally
    print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))



def basin_hopping_1d_own():
    # 1D function, own function (see above 'ownfunc')
    print('1D function, own function')
    func = ownfunc
    
    x = np.linspace(-10, 15, 1000)
    plt.plot(x, func(x))
    plt.ylim([-22, 5])
    plt.show()

    print(np.random.rand(1))

    x0=[1.0]
    # Random starting point in the range -10 ... 15
    # x0 = np.random.rand(1)*25.0 - 10.0
    print('Starting x0 =', x0)

    minimizer_kwargs = {"method": "BFGS"}
    ret = basinhopping(func, x0, stepsize=15.0, minimizer_kwargs=minimizer_kwargs,
                       niter=5, callback=print_fun)
    print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
    print("\nNote: if x0=1.0 and stepsize=default=0.5, does not\n \
          find the global minimum at about x= -6.0")
    print("      --> Set stepsize = 15.0, then most of the time OK")

    # input('press enter')



def basin_hopping_2d():
    # 2D function
    print('2D function')
    
    def func2d(x):
        f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
                  0.2) * x[0]
        df = np.zeros(2)
        df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
        df[1] = 2. * x[1] + 0.2
        return f, df

    minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    x0 = [1.0, 1.0]
    ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
                       niter=200)   # , callback=print_fun
    print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
          ret.x[1],
          ret.fun))




def main():
    
    nelder_mead()
    print('---------')  
        
    BFGS()    
    print('---------')    
    
    trust_region_version()
    print('---------')    
    
    constr_opt()
    print('---------')    

    minim_scalar()    
    print('---------')    

    basin_hopping_1d()    
    print('---------')    

    basin_hopping_1d_own()    
    print('---------')    

    basin_hopping_2d()    
    print('---------')    


    print('\nDone')

if __name__ == '__main__':
    main()



