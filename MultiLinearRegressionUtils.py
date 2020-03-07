# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:17:14 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importMultiLinearRegressionDataset(multiLinearRegressionDatasetFileName):
    
    multiLinearRegressionDataset = pd.read_csv(multiLinearRegressionDatasetFileName)
    X = multiLinearRegressionDataset.iloc[:, :-1].values
    y = multiLinearRegressionDataset.iloc[:, 1].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

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
