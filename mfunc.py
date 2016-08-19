import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from scipy.stats import linregress


def plotExamples(x,y):
# plot examples 
    import numpy as np
    import matplotlib.pyplot as plt

    m,b=np.polyfit(x,y,1)

    fig=plt.figure()
    fig.suptitle('Overall title',fontsize=14, fontweight='bold')

    ax=fig.add_subplot(2,2,1)
    ax.plot(x,y,'o',x,m*x+b,'-')
    ax.plot(x,np.ones((x.size,1)),'--')
    ax.text(1.1, 5, 'text in fig.') 
    # ax.set_yscale('log')
    ax.set_title('Curves and dots')   
    limits=(-1,11)
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    ax=fig.add_subplot(2,2,2)
    plt.hist(y,2)
    ax.set_title('Histogram')   

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
    x=np.array([1,2,3])
    y=np.array([1,5,10])
    plotExamples(x,y)





