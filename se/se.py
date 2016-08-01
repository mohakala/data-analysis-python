import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the raw data
rawdata='../../datasets/sedat.xls'
df = pd.read_excel(rawdata)



# Columns

### not needed
### df.columns = ['x', 'ref','c01','c05','c1','c15','c5','c10']

df = df.rename(columns={'Wavelength': 'wl'})
x=df['wl'].values
ref=df[0].values



# Areas and absolute difference in areas vs. ref

#areatest=np.trapz(df[0],dx=2)
#are=[0 for i in range(5)]

ar=[]
for i in df.columns[3:]:
    area=np.trapz( np.abs(df[i]-ref)    ,dx=2)
    print(i)
    ar.append(area)

conc=df.columns[3:].values  # object
concNumeric=pd.to_numeric(conc)


# Make a linear model

m,b=np.polyfit(concNumeric,ar,1) # linear
print('m,b=',m,b)



# Predictions

testSpectrum=0.3*df[0.5]+0.7*df[1.5]
area = np.trapz( np.abs(testSpectrum-ref)    ,dx=2)
predConc=(area-b)/m
print('predicted:',predConc,' calc:',0.3*0.5+0.7*1.5)

testSpectrum=0.3*df[0.5]+0.7*df[5]
area = np.trapz( np.abs(testSpectrum-ref)    ,dx=2)
predConc=(area-b)/m
print('predicted:',predConc,' calc:',0.3*0.5+0.7*5)

testSpectrum=1.0*df[0.5]
area = np.trapz( np.abs(testSpectrum-ref)    ,dx=2)
predConc=(area-b)/m
print('predicted:',predConc,' calc:',1.0*0.5)





# Visualization

fig=plt.figure()

ax=fig.add_subplot(2,2,1)
ax.plot(x,df[0],'-',x,df[10],'--')

ax=fig.add_subplot(2,2,2)
ax.plot(x,df[0.1]-ref,'-',x,df[1]-ref,'-',x,df[0.5]-ref,'-')

ax=fig.add_subplot(2,2,3)
ax.plot(conc,ar,'o',conc,m*conc+b,'k-')

ax=fig.add_subplot(2,2,4)
ax.plot(x,testSpectrum-ref,'-')



plt.show()
