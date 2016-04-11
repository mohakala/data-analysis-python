import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy import stats

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
from mfunc import *

""" Follow
http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2

A Get the data
B Exploratory analysis in Python using Pandas
    Introduction to series and dataframes
    Analytics Vidhya dataset- Loan Prediction Problem
C Data Munging in Python using Pandas
D Building a Predictive Model in Python
    Logistic Regression
    Decision Tree
    Random Forest
"""


def getData(rawdata):
    df = pd.read_csv(rawdata) 
    return df

if __name__ == '__main__':
    lin()

    # A Get the data
    rawdata='../../datasets/nurmijarvi_asunnot_250316.csv'
    #rawdata='nurmijarvi_asunnot_250316.csv'
    df=getData(rawdata)
    print("raw data dataframe size:",df.shape)


    # B Exploratory analysis    
    # B.1 Quick data exploration
    print(df.head(4))
    lin()
    print(df.describe())
    lin()
    print('Frequency distributions:')
    print(df['Kaupunginosa'].value_counts())
    print(df['Talotiedot'].value_counts())
    print(df['Kunto'].value_counts())

    # B.2 Distribution analysis
    fig=plt.figure()
    df['Vh'].hist(bins=20)
    plt.title('Vh')
    plt.show()
    if(False):
        df.boxplot(column='Vh', by ='Talotiedot')
        plt.show()
        df.boxplot(column='Vh', by ='Kaupunginosa')
        plt.show()
        df.boxplot(column='Neliohinta', by ='Kaupunginosa')
        plt.show()
        temp3 = pd.crosstab(df['Huoneet'], df['Talotiedot'])
        temp3.plot(kind='bar', stacked=True, color=['red','blue','green'], grid=False)
        plt.show()
    
    # C Data munging
    lin()
    missValues=df.apply(lambda x: sum(x.isnull()),axis=0)
    print('Missing values in columns:\n',missValues)

    


    
    
    
