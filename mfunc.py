import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress

def lin():
    print('-------------------------------------')

def pcnt(val,ref=0.0):  # return ratio+percentage values as a string    
    if ref==0.0:
        return(str(val))
    else:
        aux=int(100*val/ref*100.0)/100
#        st=str(val)+'/'+str(ref)+'='+str(val/ref*100.0)+'%'
        pcntString=str(val)+'/'+str(ref)+'='+str(aux)+'%'
        return(pcntString)

def kluvut(x):
# http://papers.mpastell.com/scipy_opas.pdf    
    keskiarvo = np.mean(x)
    keskihajonta = np.std(x, ddof=1)
    n = max(x.shape)
    keskivirhe = keskihajonta/np.sqrt(n)
    return(keskiarvo, keskihajonta, keskivirhe)

def pcntval(val,ref):
    return(val/ref*100)



