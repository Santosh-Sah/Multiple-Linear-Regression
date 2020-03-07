# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:41:21 2020

@author: Santosh Sah
"""

from MultiLinearRegressionUtils import (readMultiLinearRegressionXTest, readMultiLinearRegressionModel)

"""
test themodel on testing dataset
"""
def testMultiLinearRegressionModel():
    
    X_test = readMultiLinearRegressionXTest()
    multiLinearRegressionModel = readMultiLinearRegressionModel()
    
    y_pred = multiLinearRegressionModel.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    testMultiLinearRegressionModel()
