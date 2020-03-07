# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:01:53 2020

@author: Santosh Sah
"""

from MultiLinearRegressionUtils import importMultiLinearRegressionDataset, saveTrainingAndTestingDataset

def preprocess():
    
    X_train, X_test, y_train, y_test = importMultiLinearRegressionDataset("Multi_Linear_Regression_50_Startups.csv")
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()