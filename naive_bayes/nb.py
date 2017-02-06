
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


# Testing Naive Beyes for the Iris dataset

import pandas as pd

def exploreData(df):
    print(df.head(5))
    print(df.describe())
    print("Data types in df:\n",df.dtypes)
    #print(df[0].value_counts())

# Get data
# - no header
df = pd.read_csv('../../datasets/iris.data',header=None)

# Data munging: categorize the text
df[4] = df[4].astype('category')

# Explore
exploreData(df)
print('Categories:',df[4].cat.categories)

# Select training and testing sets
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]
print('Sizes, train, test:',len(train),len(test))
print('Test set:\n',test.head(3))

Xtrain=train[[0,1,2,3]].as_matrix()
ytrain=train[[4]].values.ravel()

#Xtrain=train[[0,1,2,3]]
#ytrain=train[[4]]


#xt=Xtrain.as_matrix()

Xtest=test[[0,1,2,3]].as_matrix()
ytest=test[[4]].values



# Train 

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets 
model.fit(Xtrain, ytrain)

# Test 

model.predict(Xtest[0])
for i in range(len(ytest)):
    predicted= model.predict(Xtest[i])
    print('pred, true:',predicted,ytest[i])










