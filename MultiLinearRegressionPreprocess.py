# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:01:53 2020

@author: Santosh Sah
"""

from MultiLinearRegressionUtils import (importMultiLinearRegressionDataset, saveTrainingAndTestingDataset, 
                                        importMultiLinearRegressionDatasetForOLS, saveDatasetForOLS, splitTrainingAndTestingSetOLS,
                                        saveTrainingAndTestingDatasetOLS)

def preprocess():
    
    X_train, X_test, y_train, y_test = importMultiLinearRegressionDataset("Multi_Linear_Regression_50_Startups.csv")
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

def olsPreprocess():
    
    X, y = importMultiLinearRegressionDatasetForOLS("Multi_Linear_Regression_50_Startups.csv")
    saveDatasetForOLS(X, y)

def olsPreprocessTrainTest():
    
    X_train_OLS, X_test_OLS, y_train_OLS, y_test_OLS = splitTrainingAndTestingSetOLS()
    saveTrainingAndTestingDatasetOLS(X_train_OLS, X_test_OLS, y_train_OLS, y_test_OLS)
    
if __name__ == "__main__":
    #preprocess()
    #olsPreprocess()
    olsPreprocessTrainTest()