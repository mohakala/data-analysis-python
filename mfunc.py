import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress
import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
#from mfunc import *  # Some own functions

def getData(filename):
    import pandas as pd
    filename='xxx'
    df = pd.read_excel(filename)
    df = pd.read_csv(filename) 
    return df


def quickStudy2(x,y):
    # Descriptive statistics and linear regrsssion
    print('MEAN X = ',x.mean(),'+_',np.std(x, ddof=1)/np.sqrt(len(x)))
    print('MEAN Y = ',y.mean(),'+_',np.std(y, ddof=1)/np.sqrt(len(x)))
    print('STD Y = ',np.std(y, ddof=1))
    slope, intercept, r, prob2, see = linregress(x, y)
    # see = standard errof of estimated gradient
    print('Pearson correlation coefficient r:',r)
    print('R^2 coefficient of determination (Selitysaste):',r**2)
    print('P-VALUE (2-SIDED) FOR SLOPE TO BE ZERO =',prob2)
    print('slope,intercept=',slope,intercept)
    m,b=np.polyfit(x,y,1) 
    print('m,b=',m,b)

    # Ordinary linear system for x,y
    xx= sm.add_constant(x,prepend=False)
    results=sm.OLS(y,xx).fit()
    print(results.summary())

    # Ordinary linear system for x,x^2,y
    xx= sm.add_constant(x,prepend=False)
    x2=x[:,np.newaxis]*x[:,np.newaxis] # second power
    xxx = np.concatenate((xx,x2),axis=1)
    results=sm.OLS(y,xxx).fit()


def wilcoxon(x,y):
    # Wilcoxon rank-sum test for pairs of samples
    # Are the samples from different or same distribution?
    import scipy
    z_stat, p_val = scipy.stats.ranksums(x,y)  
    print('P-value of ranksumtest=',p_val)


def mannWhitney(x,y):
    # Mann-Whitney
    # Are the samples from different or same distribution?
    import scipy
    z_stat, p_val = scipy.stats.mannwhitneyu(x, y, 'two-sided')
    print('P-value of Mann-Whitney=',p_val)
    

def plotExamples(x,y):
# plot examples 
    import numpy as np
    import matplotlib.pyplot as plt

# Tune markersizes
#    plt.rcParams['lines.markersize'] = 10

    
    m,b=np.polyfit(x,y,1)

    fig=plt.figure()
    fig.suptitle('Overall title',fontsize=14, fontweight='bold')

    ax=fig.add_subplot(2,2,1)
    ax.plot(x,y,'o',x,m*x+b,'-')
    # , label="Fe"
    ax.plot(x,np.ones((x.size,1)),'--')
    ax.text(1.1, 5, 'text in fig.') 
    # ax.set_yscale('log')
    ax.set_title('Curves and dots')   
    limits=(-1,11)
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    # ax.legend()
    boxAxes=False
    if(boxAxes): ax.set_aspect(1./ax.get_data_ratio())

    ax=fig.add_subplot(2,2,2)
    plt.hist(y,2)
    ax.set_title('Histogram')   

# http://matplotlib.org/users/tight_layout_guide.html    
    plt.tight_layout()
    plt.show()

 
def lin():
    print('-------------------------------------')

def pcnt(val,ref=0.0):
    # returns ratio + percentage value (two decimals) as a string    
    # Usage example: print(pcnt(8.0,12.1)) 
    if ref==0.0:
        return(str(val))
    else:
        aux=int(100*val/ref*100.0)/100
        pcntString=str(val)+'/'+str(ref)+'='+str(aux)+'%'
        return(pcntString)

def kluvut(x):
# http://papers.mpastell.com/scipy_opas.pdf    
    keskiarvo = np.mean(x)
    otoskeskihajonta = np.std(x, ddof=1)
    n = max(x.shape)
    keskivirhe = keskihajonta/np.sqrt(n)
    return(keskiarvo, otoskeskihajonta, keskivirhe)

def pcntval(val,ref):
    return(val/ref*100)


if __name__ == '__main__':
    print('Main of mfunc.py')
    # x=np.array([1,2,3])
    # y=np.array([1,5,10])
    x=np.arange(8)
    y=np.array([0,1,3,4,5,8,9,10])

    # plotExamples(x,y)
    quickStudy2(x,y)
    wilcoxon(x,y)
    mannWhitney(x,y)




