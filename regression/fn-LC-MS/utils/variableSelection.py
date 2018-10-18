from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, RandomizedLasso
from sklearn.feature_selection import RFE
from .seabornPlot import plotfeatureImportance
import pandas as pd


def randomforest_vi(X, Y, xlabel, variableNames):
    # try random forest
    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X, Y)
    df = pd.DataFrame({xlabel: variableNames, 'importance': rf.feature_importances_})
    plotfeatureImportance(df, 'Random Forest')


def randomizedLasso_vi(X, Y, xlabel, variableNames, alpha=0.025):
    # %% randomized lasso
    rlasso = RandomizedLasso()
    rlasso.fit(X, Y)
    df = pd.DataFrame({xlabel: variableNames, 'importance': rlasso.scores_})
    plotfeatureImportance(df, 'Randomlized Lasso')


def rfe_rf_vi(X, Y, xlabel, variableNames):
    # use rf as the model
    rf = RandomForestRegressor(n_estimators=100)
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(rf, n_features_to_select=1)
    rfe.fit(X, Y)
    df = pd.DataFrame({xlabel: variableNames, 'importance': len(variableNames) - rfe.ranking_})
    plotfeatureImportance(df, 'Recursive feature elimination')
