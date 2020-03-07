# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:28:23 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LinearRegression
from MultiLinearRegressionUtils import (saveMultiLinearRegressionModel, readMultiLinearRegressionXTrain, readMultiLinearRegressionYTrain)

"""
Train simple linear regression model 
"""
def trainMultiLinearRegressionModel():
    
    X_train = readMultiLinearRegressionXTrain()
    y_train = readMultiLinearRegressionYTrain()
    
    multiLinearRegression = LinearRegression()
    multiLinearRegression.fit(X_train, y_train)
    
    saveMultiLinearRegressionModel(multiLinearRegression)

if __name__ == "__main__":
    trainMultiLinearRegressionModel()