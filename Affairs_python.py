# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:41:31 2021

@author: UMANG
"""


import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
affairs = pd.read_csv(r'C:\Users\UMANG\OneDrive\Desktop\Logistic regression/Affairs.csv', sep = ",")

affairs.loc[affairs.naffairs > 0, "naffairs"] = 1

#removing CASENUM
a1 = affairs.drop('X', axis = 1)
#a1.head(11)
a1.describe()
a1.isna().sum()

# To drop NaN values
df = claimants.dropna()

# Imputating the missing values           
# Mean Imputation - CLMAGE is a continuous data

#mean_value = c1.CLMAGE.mean()
#mean_value
#c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
#c1.CLMAGE.isna().sum()

# For Median imputation try this
# median_value = claimants.CLMAGE.median()
# claimants1['CLMAGE'] = claimants1['CLMAGE'].fillna(median_value)


# For Mode - for Discrete variables
# CLMSEX
#mode_CLMSEX = c1.CLMSEX.mode()
#mode_CLMSEX
#c1.CLMSEX = c1.CLMSEX.fillna((mode_CLMSEX)[0])
#c1.CLMSEX.isna().sum()

# CLMINSUR
#mode_INSUR = c1['CLMINSUR'].mode()
#mode_INSUR
#c1['CLMINSUR'] = c1['CLMINSUR'].fillna((mode_INSUR)[0])
#c1.CLMINSUR.isna().sum()

# SEATBELT
#mode_SB = c1['SEATBELT'].mode()
#mode_SB
#c1['SEATBELT'] = c1['SEATBELT'].fillna((mode_SB)[0])
#c1.SEATBELT.isna().sum()

# Alternate approach
########## Median Imputation for all the columns ############
a1.fillna(a1.median(), inplace=True) #not requre for this data set 
a1.isna().sum() #not requre for this data set

#c1.CLMAGE.median()
#c1.CLMINSUR.median()
#############################################################

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = a1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(a1.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(a1.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
a1["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
a1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(a1["pred"], a1["naffairs"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(a1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (93 + 29)/(181) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (212 + 73)/(420)
print(accuracy_train)
