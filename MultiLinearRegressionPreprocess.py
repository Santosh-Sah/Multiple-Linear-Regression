# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:01:53 2020

@author: Santosh Sah
"""

from MultiLinearRegressionUtils import (importMultiLinearRegressionDataset, saveTrainingAndTestingDataset, 
                                        importMultiLinearRegressionDatasetForOLS, saveDatasetForOLS)

def preprocess():
    
    X_train, X_test, y_train, y_test = importMultiLinearRegressionDataset("Multi_Linear_Regression_50_Startups.csv")
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

def olsPreprocess():
    
    X, y = importMultiLinearRegressionDatasetForOLS("Multi_Linear_Regression_50_Startups.csv")
    saveDatasetForOLS(X, y)
    
if __name__ == "__main__":
    #preprocess()
    olsPreprocess()