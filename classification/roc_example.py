# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:56:02 2018

http://blog.yhat.com/posts/roc-curves.html
with modification by M.H.

"""

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


# Make synthetic data
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=5)
Xtrain = X[:9000]
Xtest = X[9000:]
ytrain = y[:9000]
ytest = y[9000:]

clf = LogisticRegression()
clf.fit(Xtrain, ytrain)


# Plot ROC curve
from sklearn import metrics
preds = clf.predict_proba(Xtest)[:,1]
fpr, tpr, tresholds = metrics.roc_curve(ytest, preds)

# fpr = 1 - specificity = 1 - TN / (TN + FP)
# fpr = FP / (TN + FP)

plt.plot(fpr, tpr, '-')


# Calculate AUC
from sklearn.metrics import roc_auc_score
y_true = ytest
y_scores = preds
print('auc:', roc_auc_score(y_true, y_scores))




