import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rawdata='../../datasets/sedat.xls'
df = pd.read_excel(rawdata)

### not needed
### df.columns = ['x', 'ref','c01','c05','c1','c15','c5','c10']

df = df.rename(columns={'Wavelength': 'wl'})

x=df['wl'].values
ref=df[0].values



# Areas and absolute difference in areas vs. ref

#areatest=np.trapz(df[0],dx=2)
#are=[0 for i in range(5)]

ar=[]
for i in df.columns[1:]:
    area=np.trapz( np.abs(df[i]-ref)    ,dx=2)
    print(i)
    ar.append(area)

conc=df.columns[1:].values



# Visualization

fig=plt.figure()

ax=fig.add_subplot(2,2,1)
ax.plot(x,df[0],'-',x,df[10],'--')

ax=fig.add_subplot(2,2,2)
ax.plot(x,df[0.1]-ref,'-',x,df[1]-ref,'-',x,df[0.5]-ref,'-')

ax=fig.add_subplot(2,2,3)
ax.plot(conc,ar,'o')


plt.show()
