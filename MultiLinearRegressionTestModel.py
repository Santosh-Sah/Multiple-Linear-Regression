# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:41:21 2020

@author: Santosh Sah
"""

from MultiLinearRegressionUtils import (readMultiLinearRegressionXTestOLS, readMultiLinearRegressionModelOLS)

"""
test themodel on testing dataset
"""
def testMultiLinearRegressionModelOLS():
    
    X_test = readMultiLinearRegressionXTestOLS()
    multiLinearRegressionModel = readMultiLinearRegressionModelOLS()
    
    y_pred = multiLinearRegressionModel.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    testMultiLinearRegressionModelOLS()
