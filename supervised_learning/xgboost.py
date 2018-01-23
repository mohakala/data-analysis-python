# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:39:01 2018

Code from:
https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

https://anaconda.org/anaconda/py-xgboost
conda install -c anaconda py-xgboost=0.60



"""

def issue_1():
    """
    ImportError: 
        cannot import name 'XGBClassifier'
    
    Details: 
        xgboost is in envs packages
        but not in pip-packages
        also not listed with conda list
    
    Workaround:
        cd C:\ProgramData\Anaconda3\envs\tradml
        import xgboost
        After that seems to work 
        
    Related links:
        https://github.com/jupyter/notebook/issues/2359
        (https://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment/46082615)
        https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en
    """
    pass


print("[comment] cd ..\..\..\PrograData... tradml")
print("[comment] import xgboost --> Then works?!")

def findPackages():
    import pip
    installed_packages = pip.get_installed_distributions()
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    print(installed_packages_list)

findPackages()
import sys
print(sys.path)
print(sys.executable)

try:
    import xgboost
    print('xgboost successfully imported:')
    print('  ', xgboost)
except:
    print('import xgboost; print(xgboost) failed')
    


# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('..\..\datasets\pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))