
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






