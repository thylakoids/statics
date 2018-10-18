from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


def OLS(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


def myOLS(X, y):
    lm = LinearRegression()
    lm.fit(X, y)
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)

    newX = np.append(np.ones((len(X), 1)), X, axis=1)
    sigma2 = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))

    var_b = sigma2 * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
                                                                                                 p_values]
    print(myDF3)

    p = len(newX[0]) - 1
    N = X.shape[0]
    RSS_tot = sum(abs((y - y.mean())**2))
    RSS1 = sum((y - predictions) ** 2)
    F = ((RSS_tot - RSS1) / p) / sigma2
    F_statistic = 1 - stats.f.cdf(F, p, N - p - 1)
    print('F-statistic:', F_statistic)

    Rsquared = 1 - RSS1 / RSS_tot
    Rsquared_adjust = 1 - (1 - Rsquared) * (N - 1) / (N - p - 1)
    print('R-squared:', Rsquared)
    print('adjust R-squared', Rsquared_adjust)

    return {'coefficient': myDF3, 'R-squared': Rsquared, 'F-statistic': F_statistic,
            'SE': RSS1, 'adjust R-squared': Rsquared_adjust}


def randomForest(X, y):
    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X, y)
    predictions = rf.predict(X)
    RSS_tot = sum(abs((y - y.mean()) ** 2))
    RSS1 = sum((y - predictions) ** 2)
    Rsquared = 1 - RSS1 / RSS_tot
    # Rsquared = rf.score(X,Y)
    return {'SE': RSS1, 'R-squared': Rsquared, 'model': rf}


def cross_val_rf(X, y):
    rf = RandomForestRegressor(n_estimators=500)
    scores = cross_val_score(rf, X, y, cv=5)
    return scores
