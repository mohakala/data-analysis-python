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


# Make a linear model for c=0.5 and above

m,b=np.polyfit(concNumeric,ar,1) # linear
print('m,b=',m,b)


arAll=[]  # All values
for i in df.columns[1:]:
    area=np.trapz( np.abs(df[i]-ref)    ,dx=2)
    print(i)
    arAll.append(area)
concAll=df.columns[1:].values  # object





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


# How well the known datapoints are predicted?

print(' ')
for i in range(5):
    predConc=(ar[i]-b)/m
    print('predicted:',predConc,' conc:',conc[i])
    




# Make a linear model with 4 datapoints out of 5
# Check then how well the known one is predicted

print(' ')
concError=[]

points=(1,2,3,4)
missing=0
aux1=list( concNumeric[i] for i in points )
aux2=list( ar[i]          for i in points )
mm,bb=np.polyfit(aux1,aux2,1) # linear
predConc=(ar[missing]-bb)/mm
print('predicted:',predConc,' known:',conc[missing])
concError.append(predConc-conc[missing])

points=(0,2,3,4)
missing=1
aux1=list( concNumeric[i] for i in points )
aux2=list( ar[i]          for i in points )
mm,bb=np.polyfit(aux1,aux2,1) # linear
predConc=(ar[missing]-bb)/mm
print('predicted:',predConc,' known:',conc[missing])
concError.append(predConc-conc[missing])

points=(0,1,3,4)
missing=2
aux1=list( concNumeric[i] for i in points )
aux2=list( ar[i]          for i in points )
mm,bb=np.polyfit(aux1,aux2,1) # linear
predConc=(ar[missing]-bb)/mm
print('predicted:',predConc,' known:',conc[missing])
concError.append(predConc-conc[missing])

points=(0,1,2,4)
missing=3
aux1=list( concNumeric[i] for i in points )
aux2=list( ar[i]          for i in points )
mm,bb=np.polyfit(aux1,aux2,1) # linear
predConc=(ar[missing]-bb)/mm
print('predicted:',predConc,' known:',conc[missing])
concError.append(predConc-conc[missing])

points=(0,1,2,3)
missing=4
aux1=list( concNumeric[i] for i in points )
aux2=list( ar[i]          for i in points )
mm,bb=np.polyfit(aux1,aux2,1) # linear
predConc=(ar[missing]-bb)/mm
print('predicted:',predConc,' known:',conc[missing])
concError.append(predConc-conc[missing])

print('concError:',concError)






# Visualization

fig=plt.figure()

ax=fig.add_subplot(2,2,1)
ax.plot(x,df[0.1]-ref,'-',x,df[0.5]-ref,'-',x,df[1]-ref,'-',x,df[1.5]-ref,'-',x,df[5]-ref,'-',x,df[10]-ref,'-')
ax.set_title("Changes in spectra wrt. 0-spectrum")
ax.axes.get_xaxis().set_ticks([])
ax.set_xlabel('Wavelength')

ax=fig.add_subplot(2,2,2)
ax.plot(x,df[0.1]-ref,'-',x,df[0.5]-ref,'-',x,df[1]-ref,'-',x,df[1.5]-ref,'-',x,df[5]-ref,'-',x,df[10]-ref,'-')
ax.set_title("Changes in spectra wrt. 0-spectrum")
#ax.axes.get_xaxis().set_ticks([])
ax.set_xlabel('Wavelength')
ax.set_xlim(1820,1890)
ax.set_ylim(-80,100)
ax.text(1835, -50, 'zooming into the region of isosbestic points')
ax.text(1835,-60, 'shows some fluctuations in the sensor data')

ax=fig.add_subplot(2,2,3)
ax.plot(concAll,arAll,'o')
#ax.set_xlim(0,10)
#ax.set_ylim(0,60000)
ax.set_title("Absolute area difference wrt. 0-spectrum")

ax=fig.add_subplot(2,2,4)
ax.plot(conc,ar,'o',conc,m*conc+b,'k-')
ax.set_title("Linear fit for concentrations c=0.5 ... 10")



plt.show()
