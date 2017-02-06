
#https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#Assign predictor and target variables
X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets 
model.fit(X, y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print(predicted)



#import pandas as pd
#person = pd.read_csv(‘example.csv’)
#mask = np.random.rand(len(sales)) < 0.8
#train = sales[mask]
#test = sales[~mask]


# Iris dataset

import pandas as pd

def getData(rawdata):
    df = pd.read_csv(rawdata) 
    return df

def exploreData(df):
    print(df.head(16))
    print(df.describe())
    print("Data types in df:\n",df.dtypes)
    print('Frequency distributions:')
    print(df.value_counts())

df = getData('../../datasets/iris.data')
exploreData(df)








