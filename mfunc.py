import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress
import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
#from mfunc import *  # Some own functions

def getData(filename):
    """
    Different ways to read a datafile
    """
    # Without pandas

    # With pandas
    import pandas as pd
    filename='xxx'
    df = pd.read_excel(filename)
    path = 'ftp://ftp....'
    path = 'C://Python34/...'
    # Read the n:th sheet of excel-file, n=0,1,...
    df = pd.read_excel(path, sheetname=1, header=None) 
      
    df = pd.read_csv(filename) 
    df = pd.read_csv(path, header=None, names=['X1', 'X2', 'Y'])

    return df


def listModules():
    # http://stackoverflow.com/questions/739993/how-can-i-get-a-list-of-locally-installed-python-modules
    import pip
    installed_packages = pip.get_installed_distributions()
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    print(installed_packages_list)

def correlationTests():
    """
    Introductino to Correlation
         https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials
    Pearson correlation coefficient
    Spearman's rank correlatino coefficient
    Kendall's tau: directional agreement (concordant pairs)
    """
    import pandas as pd
    print('Correlation tests (www.datascience.com/...)')

    # Example 1
    k = pd.DataFrame()
    k['X'] = np.arange(5)+1
    k['Y'] = [7, 5,1, 6, 9]
    print('Kendall:\n',k.corr(method='kendall'))
    print('Pearson:\n',k.corr())

    # Example 2
    path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    mpg_data = pd.read_csv(path, delim_whitespace=True, header=None,
            names = ['mpg', 'cylinders', 'displacement','horsepower',
            'weight', 'acceleration', 'model_year', 'origin', 'name'],
            na_values='?')
    mpg_data.info()

    print('Correlation between mpg and weight, Pearson:')
    print(mpg_data['mpg'].corr(mpg_data['weight']))
    # pairwise correlation
    print(mpg_data.drop(['model_year', 'origin'], axis=1).corr(method='pearson'))

    # Plot
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [16, 6]
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax=ax.flatten()
    cols = ['weight', 'horsepower', 'acceleration']
    colors=['#415952', '#f35134', '#243AB5', '#243AB5']
    j=0
    for i in ax:
        if j==0:
            i.set_ylabel('MPG')
        i.scatter(mpg_data[cols[j]], mpg_data['mpg'],  alpha=0.5, color=colors[j])
        i.set_xlabel(cols[j])
        i.set_title('Pearson: %s'%mpg_data.corr().loc[cols[j]]['mpg'].round(2)+' Spearman: %s'%mpg_data.corr(method='spearman').loc[cols[j]]['mpg'].round(2))
        j+=1
    plt.show()


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

def r2values(x,y,ypred,p):
    # http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    # Returns the values of r2 and adjusted r2
    yave=np.mean(y)
    ssreg = np.sum((ypred-yave)**2)   
    sstot = np.sum((y - yave)**2) 
    ssres = np.sum((ypred-y)**2)
    r2 = ssreg / sstot        # or: r2 = 1-(ssres/sstot))
    n=len(y) 
    r2adj = 1- ( (ssres/(n-p-1)) / (sstot/(n-1)) ) 
    return(r2,r2adj)


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


def kluvut(x):
    # http://papers.mpastell.com/scipy_opas.pdf    
    keskiarvo = np.mean(x)
    otoskeskihajonta = np.std(x, ddof=1)
    n = max(x.shape)
    keskivirhe = keskihajonta/np.sqrt(n)
    return(keskiarvo, otoskeskihajonta, keskivirhe)


def numpyExamples(x,y):
    # Numpy examples and plot
    m=np.polyfit(x,y,5)
    poly = np.poly1d(m)
    ypred=poly(x) # or:
    ypred=np.poly1d(m)(x)
    
    fig=plt.figure()
    plt.plot(x, y, 'o')    # Original data
    xx=np.linspace(np.min(x),np.max(x),50) # Dense grid. start, end, number
    plt.plot(xx, np.poly1d(m)(xx),'-')
    plt.show()

"""
##############################################################    
##### Plot examples #####    
##############################################################    
"""

def plot1(df):
    import matplotlib.pyplot as plt
    fig=plt.figure()
    plt.scatter(df[0], df[1])
    plt.plot(df[0], df[1], 'o-')
    plt.show()

# s0100 markersize:
# plt.scatter(mu[:,0], mu[:,1], s=100, c=np.unique(Y_hat))

# no ticklabels
# plt.xticks(())
# plt.yticks(())

def plotImages():
# image, 784 points, plot to wubfigures without labels
    plt.rc("image", cmap="binary") # use black/white palette for plotting
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(X_digits[i].reshape(28,28))
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()



def plotExamples(x,y):
    # Some plotting examples 
    import numpy as np
    import matplotlib.pyplot as plt

    # Tune markersizes
    # plt.rcParams['lines.markersize'] = 10
    
    m,b=np.polyfit(x,y,1)

    fig=plt.figure()
    fig.suptitle('Overall title',fontsize=14, fontweight='bold')

    # Dense grid from xmin ... xmax:
    # xx=np.linspace(np.min(x),np.max(x),50) # start, end, linspace

    ax=fig.add_subplot(2,2,1)
    ax.plot(x,y,'o',x,m*x+b,'-')
    # , label="Xx"
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

    correlationTests()
    numpyExamples(x,y)    

    # Get r2 and adjusted r2 values
    # example case
    m=np.polyfit(x,y,4) 
    poly = np.poly1d(m)
    ypred=poly(x)
    deg=len(m)-1
    r2, r2adj = r2values(x,y,ypred,deg)
    print('r2 for the fit:',r2)
    print('Adjusted r2 for the fit:',r2adj,'degree of polynomial:',deg)



