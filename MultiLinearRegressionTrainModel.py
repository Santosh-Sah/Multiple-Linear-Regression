# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:28:23 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from MultiLinearRegressionUtils import (saveMultiLinearRegressionModel, readMultiLinearRegressionXTrain, readMultiLinearRegressionYTrain,
                                        readMultiLinearRegressionModel, readDatasetOLSX, readDatasetOLSY, saveMultiLinearRegressionModelOLS,
                                        readMultiLinearRegressionModelOLS)

"""
Train multi linear regression model 
"""
def trainMultiLinearRegressionModel():
    
    X_train = readMultiLinearRegressionXTrain()
    y_train = readMultiLinearRegressionYTrain()
    
    multiLinearRegression = LinearRegression()
    multiLinearRegression.fit(X_train, y_train)
    
    saveMultiLinearRegressionModel(multiLinearRegression)

"""
Train multi linear regression model OLS
"""
def trainMultiLinearRegressionModelOLS():
    
    X_OLS = readDatasetOLSX()
    y_OLS = readDatasetOLSY()
    
    X_opt = X_OLS[:, [0, 3]]
    
    multiLinearRegressionOLS = sm.OLS(endog = y_OLS, exog = X_opt).fit()
    
    saveMultiLinearRegressionModelOLS(multiLinearRegressionOLS)

if __name__ == "__main__":
    trainMultiLinearRegressionModelOLS()