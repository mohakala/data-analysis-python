#https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/

print('--- Testing the basics ---')

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


# Quick way to get training and testing sets 
#import pandas as pd
#person = pd.read_csv(‘example.csv’)
#mask = np.random.rand(len(sales)) < 0.8
#train = sales[mask]
#test = sales[~mask]


##########################################
# Testing Naive Beyes for the Iris dataset
##########################################

print('--- Testing for the Iris dataset ---')

import pandas as pd

def exploreData(df):
    print(df.head(5))
    print(df.describe())
    print("Data types in df:\n",df.dtypes)

# Get data (datafile has no header)
df = pd.read_csv('../../datasets/iris.data',header=None)

# Data munging: categorize the text 
# https://www.continuum.io/content/pandas-categoricals
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

Xtest=test[[0,1,2,3]].as_matrix()
ytest=test[[4]].values

# Train 
model = GaussianNB() #Create a Gaussian Classifier
model.fit(Xtrain, ytrain)


# Test 
correct=0
for i in range(len(ytest)):
    predicted= model.predict(Xtest[i])
    print('pred, true:',predicted,ytest[i])
    if(predicted==ytest[i]): correct+=1
print('Correct predictions:',correct/len(ytest)*100.0,'%')

print('Other tetss, predict_proba (probability to belong to a class):')
print(model.predict_proba(Xtest[0]))
print('Categories:',df[4].cat.categories)

# Cross-validation
# https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
# from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
n_folds = 10
score=cross_val_score(model, Xtrain, ytrain, cv=n_folds, n_jobs=1)
print('n_folds, average cross-validation score:',n_folds, score.mean())



##########################################
# Naive Beyes articles
##########################################

# https://www.quora.com/What-types-of-data-sets-are-appropriate-for-Naive-Bayes







