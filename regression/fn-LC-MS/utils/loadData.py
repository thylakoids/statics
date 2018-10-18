import pandas as pd
import sys
import numpy as np


def load_chromatographyData(filePath):
    data = pd.read_excel(filePath, index_col=0, header=0)
    short_columns = list(np.array(data.columns[:-1]).astype(np.double).round(1))
    short_columns.append(data.columns[-1])
    data.columns = short_columns
    X = data.values[:, :-1]
    X[np.isnan(X)] = 0
    Y = data.values[:, -1]
    variableNames = data.columns[:-1]
    sampleNames = data.index
    return {'dataFrame': data, 'X': X, 'Y': Y, 'variableNames': variableNames, 'sampleNames': sampleNames}


if __name__ == '__main__':
    filePath = 'C:/Users/clay/Dropbox/statics/regression/data/chromatographydata_fn_2.xlsx'
    load_chromatographyData(filePath)
