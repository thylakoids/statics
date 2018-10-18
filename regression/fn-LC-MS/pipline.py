
from sklearn.linear_model import RandomizedLasso, Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
import pandas as pd

from config import conf
from utils.loadData import load_chromatographyData
from utils.seabornPlot import plotcorrelationMatrix, plotfeatureImportance
from utils.variableSelection import randomforest_vi, randomizedLasso_vi, rfe_rf_vi
from utils.regression import randomForest, OLS, myOLS, cross_val_rf


# %% Load data
chromatography = load_chromatographyData(conf.chromatographyDataPath)
X = chromatography['X']
Y = chromatography['Y']
variableNames = chromatography['variableNames']
sampleNames = chromatography['sampleNames']

# 0. preprocess
min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)


# prepare train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaler, Y, test_size=0.2, random_state=10)


# 1. plot data correlation, pearson correlation
plotcorrelationMatrix(chromatography['dataFrame'])

# 2. feature selection by lasso(Warning DEPRECATED)
# Sklearn implements stability selection in the randomized lasso and randomized logistics regression classes.
# randomizedLasso_vi(X, Y, 'peaks', variableNames)
# randomforest_vi(X, Y, 'peaks', variableNames)
# rfe_rf_vi(X, Y, 'peaks', variableNames)

# 3. regression
result = randomForest(X_train, y_train)
print(result['model'].score(X_test, y_test))

print(cross_val_rf(X, Y).mean())
