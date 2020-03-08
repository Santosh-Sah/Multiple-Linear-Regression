# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 09:54:37 2020

@author: Santosh Sah
"""

import statsmodels.formula.api as sm
from MultiLinearRegressionUtils import (readDatasetOLSX, readDatasetOLSY)

"""
Train multi linear regression model OLS
"""
def trainMultiLinearRegressionModelOLS():
    
    X_OLS = readDatasetOLSX()
    y_OLS = readDatasetOLSY()
    
    #X_opt = X_OLS[:, [0, 1, 2, 3, 4, 5]]
    
    """
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.951
    Model:                            OLS   Adj. R-squared:                  0.945
    Method:                 Least Squares   F-statistic:                     169.9
    Date:                Sun, 08 Mar 2020   Prob (F-statistic):           1.34e-27
    Time:                        09:38:39   Log-Likelihood:                -525.38
    No. Observations:                  50   AIC:                             1063.
    Df Residuals:                      44   BIC:                             1074.
    Df Model:                           5
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
    x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
    x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
    x3             0.8060      0.046     17.369      0.000       0.712       0.900
    x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
    x5             0.0270      0.017      1.574      0.123      -0.008       0.062
    ==============================================================================
    Omnibus:                       14.782   Durbin-Watson:                   1.283
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.266
    Skew:                          -0.948   Prob(JB):                     2.41e-05
    Kurtosis:                       5.572   Cond. No.                     1.45e+06
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.45e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    """
    """
    vatiable x2 has high p-value. remove it and create new model
    
    """
    #X_opt = X_OLS[:, [0, 1, 3, 4, 5]]
    
    
    """
                                    OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.951
    Model:                            OLS   Adj. R-squared:                  0.946
    Method:                 Least Squares   F-statistic:                     217.2
    Date:                Sun, 08 Mar 2020   Prob (F-statistic):           8.49e-29
    Time:                        10:03:14   Log-Likelihood:                -525.38
    No. Observations:                  50   AIC:                             1061.
    Df Residuals:                      45   BIC:                             1070.
    Df Model:                           4
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
    x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
    x2             0.8060      0.046     17.606      0.000       0.714       0.898
    x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
    x4             0.0270      0.017      1.592      0.118      -0.007       0.061
    ==============================================================================
    Omnibus:                       14.758   Durbin-Watson:                   1.282
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.172
    Skew:                          -0.948   Prob(JB):                     2.53e-05
    Kurtosis:                       5.563   Cond. No.                     1.40e+06
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.4e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    """
    """
    variable x1 has high p-value. remove it make new model
    
    """
    
    #X_opt = X_OLS[:, [0, 3, 4, 5]]
    
    """
                                    OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.951
    Model:                            OLS   Adj. R-squared:                  0.948
    Method:                 Least Squares   F-statistic:                     296.0
    Date:                Sun, 08 Mar 2020   Prob (F-statistic):           4.53e-30
    Time:                        10:12:46   Log-Likelihood:                -525.39
    No. Observations:                  50   AIC:                             1059.
    Df Residuals:                      46   BIC:                             1066.
    Df Model:                           3
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
    x1             0.8057      0.045     17.846      0.000       0.715       0.897
    x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
    x3             0.0272      0.016      1.655      0.105      -0.006       0.060
    ==============================================================================
    Omnibus:                       14.838   Durbin-Watson:                   1.282
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
    Skew:                          -0.949   Prob(JB):                     2.21e-05
    Kurtosis:                       5.586   Cond. No.                     1.40e+06
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.4e+06. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    """
    
    """
    variable x2 has high p-value. remove it and make the model again
    
    """
    #X_opt = X_OLS[:, [0, 3, 5]]
    
    """
                                    OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.950
    Model:                            OLS   Adj. R-squared:                  0.948
    Method:                 Least Squares   F-statistic:                     450.8
    Date:                Sun, 08 Mar 2020   Prob (F-statistic):           2.16e-31
    Time:                        10:17:54   Log-Likelihood:                -525.54
    No. Observations:                  50   AIC:                             1057.
    Df Residuals:                      47   BIC:                             1063.
    Df Model:                           2
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
    x1             0.7966      0.041     19.266      0.000       0.713       0.880
    x2             0.0299      0.016      1.927      0.060      -0.001       0.061
    ==============================================================================
    Omnibus:                       14.677   Durbin-Watson:                   1.257
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
    Skew:                          -0.939   Prob(JB):                     2.54e-05
    Kurtosis:                       5.575   Cond. No.                     5.32e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.32e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    """
    """
    variable x2 has high p-value. remove it and make new model
    
    """
    X_opt = X_OLS[:, [0, 3]]
    
    """
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.947
    Model:                            OLS   Adj. R-squared:                  0.945
    Method:                 Least Squares   F-statistic:                     849.8
    Date:                Sun, 08 Mar 2020   Prob (F-statistic):           3.50e-32
    Time:                        10:20:56   Log-Likelihood:                -527.44
    No. Observations:                  50   AIC:                             1059.
    Df Residuals:                      48   BIC:                             1063.
    Df Model:                           1
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
    x1             0.8543      0.029     29.151      0.000       0.795       0.913
    ==============================================================================
    Omnibus:                       13.727   Durbin-Watson:                   1.116
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
    Skew:                          -0.911   Prob(JB):                     9.44e-05
    Kurtosis:                       5.361   Cond. No.                     1.65e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.65e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    """
    """
    This is final model. Out of all the variable R&D Spend has lowest p-value. This is considered as the most significant variables.
    
    """
    multiLinearRegressionOLS = sm.OLS(endog = y_OLS, exog = X_opt).fit()
    
    print(multiLinearRegressionOLS.summary())
    
if __name__ == "__main__":
    trainMultiLinearRegressionModelOLS()
    