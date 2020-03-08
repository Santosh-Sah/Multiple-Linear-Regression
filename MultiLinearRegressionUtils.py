# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:17:14 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importMultiLinearRegressionDataset(multiLinearRegressionDatasetFileName):
    
    multiLinearRegressionDataset = pd.read_csv(multiLinearRegressionDatasetFileName)
    X = multiLinearRegressionDataset.iloc[:, :-1].values
    y = multiLinearRegressionDataset.iloc[:, 4].values
    
    X = multiLinearRegressionOneHotEncoder(X)
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Import dataset and read specific column.
"""
def importMultiLinearRegressionDatasetForOLS(multiLinearRegressionDatasetFileName):
    
    multiLinearRegressionDatasetOLS = pd.read_csv(multiLinearRegressionDatasetFileName)
    
    X = multiLinearRegressionDatasetOLS.iloc[:, :-1].values
    y = multiLinearRegressionDatasetOLS.iloc[:, 4].values
    
    X = multiLinearRegressionOneHotEncoder(X)
    
    X = multiLinearRegressionYIntercept(X)
    
    return X, y

"""
Encoding categorical variables
"""
def multiLinearRegressionOneHotEncoder(X):
    
    multiLinearRegressionLabelEncoder = LabelEncoder()
    X[:, 3] = multiLinearRegressionLabelEncoder.fit_transform(X[:, 3])
    
    multiLinearRegressionOneHotEncoder = OneHotEncoder(categorical_features= [3])
    X = multiLinearRegressionOneHotEncoder.fit_transform(X).toarray()
    
    #avoiding dummy variables trap
    X = X[:, 1:]
    
    return X    

"""
y = b0 + b1x1 + b2x2 + .... + bnxn
Here we are creating a data frame so that the columns present in the equation which is mentioned above.
(50, 1) dataset has has 50 rows
axis = 1 defines that we are added a column in dataset X
"""
def multiLinearRegressionYIntercept(X):
    
    X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
    
    return X

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveMultiLinearRegressionStandardScaler():
    
    multiLinearRegressionStandardScalar = StandardScaler()
    
    #Write SimpleLinearRegressionStandardScaler in a picke file
    with open("MultiLinearRegressionStandardScaler.pkl",'wb') as MultiLinearRegressionStandardScaler_Pickle:
        pickle.dump(multiLinearRegressionStandardScalar, MultiLinearRegressionStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save MultiLinearRegressionModel as a pickle file.
"""
def saveMultiLinearRegressionModel(multiLinearRegressionModel):
    
    #Write MultiLinearRegressionModel as a picke file
    with open("MultiLinearRegressionModel.pkl",'wb') as MultiLinearRegressionModel_Pickle:
        pickle.dump(multiLinearRegressionModel, MultiLinearRegressionModel_Pickle, protocol = 2)

"""
read MultiLinearRegressionStandardScaler from pickel file
"""
def readMultiLinearRegressionStandardScaler():
    
    #load MultiLinearRegressionStandardScaler object
    with open("MultiLinearRegressionStandardScaler.pkl","rb") as MultiLinearRegressionStandardScaler:
        multiLinearRegressionStandardScalar = pickle.load(MultiLinearRegressionStandardScaler)
    
    return multiLinearRegressionStandardScalar

"""
read MultiLinearRegressionModel from pickle file
"""
def readMultiLinearRegressionModel():
    
    #load MultiLinearRegressionModel model
    with open("MultiLinearRegressionModel.pkl","rb") as MultiLinearRegressionModel:
        multiLinearRegressionModel = pickle.load(MultiLinearRegressionModel)
    
    return multiLinearRegressionModel

"""
read X_train from pickle file
"""
def readMultiLinearRegressionXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readMultiLinearRegressionXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readMultiLinearRegressionYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readMultiLinearRegressionYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
Save dataset OLS regression
"""
def saveDatasetForOLS(X, y):
    
    #Write X in a picke file
    with open("X_OLS.pkl",'wb') as X_OLS_Pickle:
        pickle.dump(X, X_OLS_Pickle, protocol = 2)
    
    #Write y in a picke file
    with open("y_OLS.pkl",'wb') as y_OLS_Pickle:
        pickle.dump(y, y_OLS_Pickle, protocol = 2)

"""
read OLS dataset X from pickle file
"""
def readDatasetOLSX():
    
    #load X
    with open("X_OLS.pkl","rb") as X_OLS_pickle:
        X_OLS = pickle.load(X_OLS_pickle)
    
    return X_OLS

"""
read OLS dataset y from pickle file
"""
def readDatasetOLSY():
    
    #load y
    with open("y_OLS.pkl","rb") as y_OLS_pickle:
        y_OLS = pickle.load(y_OLS_pickle)
    
    return y_OLS

"""
Save MultiLinearRegressionModelOLS as a pickle file.
"""
def saveMultiLinearRegressionModelOLS(multiLinearRegressionModelOLS):
    
    #Write MultiLinearRegressionModelOLS as a picke file
    with open("MultiLinearRegressionModelOLS.pkl",'wb') as MultiLinearRegressionModelOLS_Pickle:
        pickle.dump(multiLinearRegressionModelOLS, MultiLinearRegressionModelOLS_Pickle, protocol = 2)

"""
read MultiLinearRegressionModelOLS from pickle file
"""
def readMultiLinearRegressionModelOLS():
    
    #load MultiLinearRegressionModelOLS model
    with open("MultiLinearRegressionModelOLS.pkl","rb") as MultiLinearRegressionModelOLS:
        multiLinearRegressionModelOLS = pickle.load(MultiLinearRegressionModelOLS)
    
    return multiLinearRegressionModelOLS
