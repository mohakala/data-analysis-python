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
from sklearn.cluster import KMeans
from random import randint


""" 
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

def dataPreparation():
    # Get rawdata, clean, handle error/missing values

    # Get rawdata
    rawdata='../../datasets/nurmijarvi_asunnot_250316.csv'
    df = pd.read_csv(rawdata)
    print("raw data dataframe size:",df.shape)

    print(df.head(2))

    # Quick data exploration
    lin()
    print(df.describe())
    lin()
    print("Data types in df:\n",df.dtypes)
    lin()
    print('Frequency distributions:')
    print(df['Kaupunginosa'].value_counts())
    print(df['Talotiedot'].value_counts())
    print(df['Kunto'].value_counts())
    lin()

    # Visualize

    # Clean: Drop useless columns
    df = df.drop('Unnamed: 11', 1)
    df = df.drop('Huoneisto', 1)
    df = df.drop('Krs', 1)
    print(df.head(2))

    # Missing values
    missValues=df.apply(lambda x: sum(x.isnull()),axis=0)
    print('Missing values in columns:\n',missValues)

    # Drop instances with missing values
    dropMissingValues=False
    if (dropMissingValues):
        df = df.dropna()
        missValues=df.apply(lambda x: sum(x.isnull()),axis=0)
        print('Missing values in columns:\n',missValues)

    # Fill categorical missing value in 'Kaupunginosa' with most frequent value
    # - help from http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
    fillWithFrequent=True
    if (fillWithFrequent):
        fill = df['Kaupunginosa'].value_counts().index[0] # most frequent value
        df['Kaupunginosa'] = df['Kaupunginosa'].fillna(fill)

    print('Frequency distributions:')
    print(df['Kaupunginosa'].value_counts())


    # Function to show the missing values in the columns
    def showMissVal(df):
        missValues=df.apply(lambda x: sum(x.isnull()),axis=0)
        if (sum(missValues)>0):
            print('Missing values in columns:\n',missValues)
        else:
            print('No missing values in the columns')
               

    showMissVal(df)
    print("dataframe size:",df.shape)

    return(df)


def dataMunging(df):
    # Transform, format, split
    return(df)


def modelData(df):
    pass



if __name__ == '__main__':
    df = dataPreparation()     # Get pandas dataframe
    dataMunging(df)
    modelData(df)





#Generic function for making a classification model and accessing performance:
#Source: http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
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







# DUMP


if __name__ == 'DON"T EXECUTE THIS, COPY-PASTE':
    lin()

    # A Get the data
    rawdata='../../datasets/nurmijarvi_asunnot_250316.csv'
    #rawdata='nurmijarvi_asunnot_250316.csv'
    df=getData(rawdata)
    print("raw data dataframe size:",df.shape)


    # B Exploratory analysis    
    # B.1 Quick data exploration
    print(df.head(16))
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
    # Manual replace
    # Better ways: learn from http://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/

    # C.3 Extreme values (not used)
    df['Vh_log'] = np.log(df['Vh'])
    df['Vh_log'].hist(bins=20)
    #plt.show()

    
    # D Predictive model

    # D.0 Auxiliary dataframe to encode categorical values to numeric
    # in a different way. Replace ok --> xok so the numerical sequence
    # corresponds to kt, rt, xok
    # Added on 4.6.2016
    df_aux=df.copy()
    df_aux['Talotiedot']=df_aux['Talotiedot'].replace({'ok': 'xok'})
#    sys.exit('Stop')
    
    # D.1 Encode categorical values to numeric
    var_mod = ['Kaupunginosa','Huoneisto','Talotiedot','Hissi','Kunto']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
        df_aux[i] = le.fit_transform(df_aux[i])
    print('new df types:\n',df.dtypes) 
    # what value corresponds to what category?
    TalotiedotName='kt','ot','rt'
    TalotiedotName3='kt','rt','xot'
    # KaupunginosaName='Alppila','Kirkonkylä','Klaukkala',...,'Rajamäki=10'

    # print(df.head(16))
    

    # Write the numerical tables back to file
    if(False):
        columns_to_file = ['Huoneet','Talotiedot','m2','Vh','Neliohinta','Rv']
        data_to_file=df[columns_to_file].values
        np.savetxt('asunnot_250316_cleaned_numerical.csv', data_to_file, delimiter=',') 
    if(False):
        # This file contains also the factor 'Kaupunginosa' 4.6.2013
        columns_to_file = ['Kaupunginosa','Huoneet','Talotiedot','m2','Vh','Neliohinta','Rv']
        data_to_file=df[columns_to_file].values
        np.savetxt('asunnot_250316_cleaned_numerical2.csv', data_to_file, delimiter=',') 
    if(False):
        # This file now contains numerical Talotiedot in the order: kt,rt,xok
        columns_to_file = ['Kaupunginosa','Huoneet','Talotiedot','m2','Vh','Neliohinta','Rv']       
        data_to_file=df_aux[columns_to_file].values
        np.savetxt('asunnot_250316_cleaned_numerical3.csv', data_to_file, delimiter=',') 
        # sys.exit('Stop')

    


    #
    # Task and features
    #
    # Features:
    # Kaupunginosa,Huoneet,Talotiedot,
    # m2,Vh,Neliohinta,Rv,Hissi,Kunto
    #
    # Task: Määritä talotieto (kt,ot,rt) annetun muuttujan perusteella.
    #

    outcome_var = 'Talotiedot'

    # D.2 Logistic regression
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
    lin()
    
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
        print('Pred: m2',i,'Talotiedot:',TalotiedotName[model.predict(i)])

    # D.2.5 Ennuste Talotiedot <-- m2,Rv
    predictor_var = ['m2','Rv']
    classification_model(model, df,predictor_var,outcome_var)


    # D.3 Logistic regression, Decision Tree, Random Forest
    lin()

    print('LogisticRegression')
    model = LogisticRegression()
    predictor_var = ['Huoneet','m2','Kaupunginosa','Rv','Hissi','Vh','Neliohinta']
#    predictor_var = ['Huoneet','m2']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
    lin()

    model=DecisionTreeClassifier()
    print('DecisionTreeClassifier')
    predictor_var = ['Huoneet','m2','Kaupunginosa','Rv','Hissi','Vh','Neliohinta']
#    predictor_var = ['Huoneet','m2','Kaupunginosa']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
    lin()

    print('RandomForestClassifier')
    model = RandomForestClassifier(n_estimators=100)
    predictor_var = ['Huoneet','m2','Kaupunginosa','Rv','Hissi','Vh']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
    lin()

    print('RandomForestClassifier')
    model = RandomForestClassifier(n_estimators=100)
    predictor_var = ['Huoneet','m2','Kaupunginosa','Rv','Hissi','Vh','Neliohinta']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)

    #Create a series with feature importances:
    featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
    print(featimp)
    lin()

    print('RandomForestClassifier')
    model = RandomForestClassifier(n_estimators=100)
    predictor_var = ['m2','Rv','Hissi','Vh']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
    lin()

    print('RandomForestClassifier')
    model = RandomForestClassifier(n_estimators=25,min_samples_split=25,max_depth=7,max_features=1)
#    predictor_var = ['m2','Vh','Hissi','Rv','Neliohinta']
    predictor_var = ['m2','Vh','Hissi','Rv']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
    lin()

    print('For previous model, sample RFs parameters and see the CV score')
    i=0
    while i < 1: # 20
        i=i+1
        ne=10*randint(1,10)
        mss=randint(2,35)
        md=randint(3,10)
        mf=randint(1,3)
        model = RandomForestClassifier(n_estimators=ne,min_samples_split=mss,max_depth=md,max_features=mf)
        predictor_var = ['m2','Vh','Hissi','Rv']
        classification_model(model, df,predictor_var,outcome_var)
        print('Params=',ne,mss,md,mf)
    lin()

    # Best param found for now: 80 18 10 1: cv=85.259%, acc=89.877
    print('Best for now:')
    ne,mss,md,mf=80, 18, 10, 1
    print('RandomForestClassifier')
    model = RandomForestClassifier(n_estimators=ne,min_samples_split=mss,max_depth=md,max_features=mf)
    predictor_var = ['m2','Vh','Hissi','Rv']
    print(predictor_var)
    classification_model(model, df,predictor_var,outcome_var)
  
    # Make predictions:
    features=(86,170000,0,1987)
    prediction=model.predict(features)
    print('Predict:',features,' Talotiedot:',TalotiedotName[prediction])
    lin()


    # D.4 K-Means 
    # First tests for K-Means
    # http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/

    print('K-Means Clustering')
    predictor_var = ['Huoneet','Talotiedot','m2','Vh','Neliohinta','Rv']

    ndf=len(df)
    upperlimit=int(0.80*ndf) # Take 80% of values for training set
    trainRange=np.arange(0,upperlimit)
    testRange=np.arange(upperlimit+1,ndf)

    # How many clusters one should choose?
    # Study inertia as a function of n_clusters
    inertiaValues = []
    scoreValues = []
    for i in range(3,14):
        k_means = KMeans(n_clusters=i, random_state=0)
        k_means.fit(df[predictor_var].iloc[trainRange,:])   #iloc[0:290,:])
        score=k_means.score(df[predictor_var].iloc[trainRange,:])
        print('i, inertia, score:',i,k_means.inertia_,score)
        inertiaValues.append(k_means.inertia_)
        scoreValues.append(k_means.score(df[predictor_var].iloc[trainRange,:]))
    if(False):
        fig=plt.figure()
        ax=fig.add_subplot(2,2,1)
        ax.plot(range(3,14),inertiaValues)
        plt.show()

    # Study the case n_clusters
    k_means = KMeans(n_clusters=3, random_state=0)
    k_means.fit(df[predictor_var].iloc[trainRange,:])
    if(False): print('Cluster centers:',k_means.cluster_centers_)
    
    # Do some classifications for the test set
    predicted = k_means.predict(df[predictor_var].iloc[testRange,:])
    print('DF:',df[predictor_var][upperlimit+1:upperlimit+5])
    print('Predicted classes:',predicted[0:4])


    # D.5 Dimensionality reduction 
    # First tests

    from sklearn import decomposition

    pca= decomposition.PCA()
    # fa= decomposition.FactorAnalysis()
    train=df[predictor_var].iloc[trainRange,:]
    train_reduced = pca.fit_transform(train)
    # ??? To study 
    
    
    # To continue...

    


    

    
        




    
