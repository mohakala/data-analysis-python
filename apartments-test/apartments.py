import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy import stats

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')
from mfunc import *

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

""" Follow
http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2

# We have:
# Kaupunginosa,Huoneet,Talotiedot(kt,ot,rt),
# m2,Vh,Neliohinta,Rv,Hissi,Kunto

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

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
    
        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]
    
        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
        #Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome]) 


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
    print("Data types in df:\n",df.dtypes)
    lin()
    print('Frequency distributions:')
    print(df['Kaupunginosa'].value_counts())
    print(df['Talotiedot'].value_counts())
    print(df['Kunto'].value_counts())

    # B.2 Distribution analysis
    fig=plt.figure()
    if(False):
        df['Vh'].hist(bins=20)
        plt.title('Vh')
        #plt.show()
        df['Neliohinta'].hist(bins=20)
        plt.title('Neliohinta')
        #plt.show()
        temp3 = pd.crosstab(df['Huoneet'], df['Kunto'])
        temp3.plot(kind='bar', stacked=True, color=['red','blue','green'], grid=False)
        plt.show()

    if(False):
        df['Vh'].hist(bins=20)
        plt.title('Vh')
        plt.show()
        df.boxplot(column='Vh', by ='Talotiedot')
        plt.show()
        df.boxplot(column='Vh', by ='Kaupunginosa')
        plt.show()
        df.boxplot(column='Neliohinta', by ='Kaupunginosa')
        plt.show()
        temp3 = pd.crosstab(df['Huoneet'], df['Talotiedot'])
        temp3.plot(kind='bar', stacked=True, color=['red','blue','green'], grid=False)
        plt.show()
        temp3 = pd.crosstab([df['Kunto'], df['Talotiedot']], df['Huoneet'] )
        temp3.plot(kind='bar', stacked=True, color=['red','blue','green','cyan','yellow','black'], grid=False)
        plt.show()
    
    # C Data munging
    # C.1 Missing values in dataset
    lin()
    missValues=df.apply(lambda x: sum(x.isnull()),axis=0)
    print('Missing values in columns:\n',missValues)

    # C.2 Fill missing values
    # df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    # Here simply replace by general 'Nurmijarvi'
    # Generally this needs to be analyzed thoroughly
    df['Kaupunginosa'].fillna('Nurmijarvi', inplace=True)

    # C.3 Extreme values
    df['Vh_log'] = np.log(df['Vh'])
    df['Vh_log'].hist(bins=20)
    #plt.show()

    
    # D Predictive model
    
    # D.1 Encode categorical values to numeric
    var_mod = ['Kaupunginosa','Huoneisto','Talotiedot','Hissi','Kunto']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    print('new df types:\n',df.dtypes) 
    # what value corresponds to what category?


    # D.2 Logistic regression

    # We have:
    # Kaupunginosa,Huoneet,Talotiedot,
    # m2,Vh,Neliohinta,Rv,Hissi,Kunto

    # Kysymys: Mikä määrää talotiedon: kt,ot,rt?
    outcome_var = 'Talotiedot'
    model = LogisticRegression()

    # D.2.1 Ennuste Talotiedot (kt,ot,rt) <-- Huoneet  
    if(False):
        temp3 = pd.crosstab(df['Huoneet'], df['Talotiedot'])
        temp3.plot(kind='bar', stacked=True, color=['red','blue','green'], grid=False)
        plt.show()

    predictor_var = ['Huoneet']
    classification_model(model, df,predictor_var,outcome_var)
    for i in range(1,5):
        print('Pred: Huoneet:',i,'Talotiedot:',model.predict(i))

    # D.2.2 Ennuste Talotiedot <-- Kaupunginosa
    predictor_var = ['Kaupunginosa']
    classification_model(model, df,predictor_var,outcome_var)
    print('Pred: Klaukkala=2','Talotiedot:',model.predict(2))
    print('Pred: Nurmijarvi=1','Talotiedot:',model.predict(1))

    # D.2.3 Ennuste Talotiedot <-- Hissi
    predictor_var = ['Hissi']
    classification_model(model, df,predictor_var,outcome_var)
    print('Pred: Hissi on=1','Talotiedot:',model.predict(1))
    print('Pred: Hissi ei=0','Talotiedot:',model.predict(0))

    # D.2.4 Ennuste Talotiedot <-- m2
    predictor_var = ['m2']
    classification_model(model, df,predictor_var,outcome_var)
    for i in range(30,150,20):
        print('Pred: m2',i,'Talotiedot:',model.predict(i))

    # D.2.5 Ennuste Talotiedot <-- m2,Rv
    predictor_var = ['m2','Rv']
    classification_model(model, df,predictor_var,outcome_var)

    

    
