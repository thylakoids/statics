import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoCV, LassoLarsCV, LassoLarsIC,RandomizedLasso
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans

from config import conf
from utils import regression,variableSelection
from utils.loadData import  load_chromatographyData
from utils.correlationOfPredictors import  plotcorrelationMatrix
from utils import lassoPath

#%% Load data
chromatography = load_chromatographyData(conf.chromatographyDataPath)
X = chromatography['X']
Y = chromatography['Y']
variableNames = chromatography['variableNames']
sampleNames = chromatography['sampleNames']

#%% preprocess
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scaler = min_max_scaler.fit_transform(X)


#%%  prepare train and test data
lr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# #%% OLS
lr.fit(X_train,y_train)
# the below expression are equal only in OLS
print 'coefficient of determination of lr in train:', lr.score(X_train,y_train)
print 'coef of lr in train:', np.corrcoef(y_train,lr.predict(X_train))[0,1]**2

print 'coefficient of determination of lr in test:', lr.score(X_test, y_test)

# #%% ridge
# scores=[]
# for alpha in np.linspace(0,10000,20):
#     ridge = Ridge(alpha = alpha)
#     ridge.fit(X_train, y_train)
#     scores.append(np.corrcoef(y_test,ridge.predict(X_test))[0,1])
# plt.plot(np.linspace(0,10000,20),scores)
# plt.show()
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
print 'coefficient of determination of ridge in train:', ridge.score(X_train,y_train)
print 'coefficient of determination of ridge in test:', ridge.score(X_test,y_test)


#%% rf
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print 'coefficient of determination of rf in train:', rf.score(X_train,y_train)
print 'coefficient of determination of rf in test:', rf.score(X_test,y_test)
# print np.corrcoef(y_test,ridge.predict(X_test))[0,1]**2


# #%% 10-folder cross validation for ridge
# scores=[]
# for alpha in np.linspace(0,10,200):
#     ridge = Ridge(alpha = alpha,normalize=True)
#     cv = ShuffleSplit(n_splits=10)
#     scores.append(cross_val_score(ridge,X,Y,cv=cv).mean())
# plt.plot(np.linspace(0,10,200),scores)
# plt.show()
# scores

# #%% 10-folder cross validation for lasso
# scores=[]
# for alpha in np.linspace(0,0.5,20):
#     lasso = Lasso(alpha = alpha,normalize=True)
#     cv = ShuffleSplit(n_splits=10)
#     scores.append(cross_val_score(lasso,X,Y,cv=cv).mean())
# plt.plot(np.linspace(0,0.5,20),scores)
# plt.show()
# scores
# # %%lasso path
# lassoPath.lARic(X,Y)
# lassoPath.lassolarscv(X,Y)

# #%% PCR
# scores=[]
# lr = LinearRegression()
# for n in range(1,25):
#     pca = PCA(n_components=n)
#     X_pca = pca.fit_transform(X)
#     cv = ShuffleSplit(n_splits=10)
#     scores.append(cross_val_score(lr,X_pca,Y,cv=cv).mean())
# plt.plot(range(1,25),scores)
# plt.show()
# scores

# #%% PLS
# scores=[]
# for n in range(1,X.shape[1]):
#     pls = PLSRegression(n_components=n)
#     cv = ShuffleSplit(n_splits=10)
#     scores.append(cross_val_score(pls,X,Y,cv=cv).mean())
# plt.plot(range(1,X.shape[1]),scores)
# plt.show()
# scores

PLS = PLSRegression(n_components = 15)
PLS.fit(X_train, y_train)
print 'coefficient of determination of PLS in train:', PLS.score(X_train,y_train)
print 'coefficient of determination of PLS in test:', PLS.score(X_test,y_test)

#%% vi
# variableSelection.randomizedLasso_vi(X,Y,range(1,25))
# variableSelection.randomforest_vi(X,Y,range(1,25))
