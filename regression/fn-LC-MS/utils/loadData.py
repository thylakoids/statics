import pandas as pd
import numpy as np
import re


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


def load_lingzhiData(filePath_x, filePath_y, sheet_name_cellCV)->pd.DataFrame:
    # 液质数据，19个标准品， 121个样品
    data_x = pd.read_excel(filePath_x, sheet_name='1', index_col=0)

    # convert 样品4号 -> 4
    data_y = pd.read_excel(filePath_y, sheet_name=sheet_name_cellCV, index_col=0)
    index_new = [int(re.findall('[0-9]+', str(samplename))[0]) for samplename in data_y.index]
    data_y.index = index_new

    # merge them，细胞实验缺少63号， 1-4有120数据， 5-7有119数据
    data = pd.merge(data_x, data_y, how='inner', left_index=True, right_index=True)
    X = data.values[:, :-1]
    Y = data.values[:, -1]
    Y[Y < 0] = 0
    Y[Y > 100] = 100
    variableNames = data.columns[:-1]
    sampleNames = data.index
    return {'dataFrame': data, 'X': X, 'Y': Y, 'variableNames': variableNames, 'sampleNames': sampleNames}


if __name__ == '__main__':
    filePath = '/Users/dracarys/Dropbox/statistics/regression/data/chromatographydata_fn_2.xlsx'
    lingzhi_x_Path = '/Users/dracarys/Dropbox/statistics/regression/data/lingzhi_x.xlsx'
    lingzhi_y_Path = '/Users/dracarys/Dropbox/statistics/regression/data/lingzhi_y.xlsx'
    load_chromatographyData(filePath)
    data = load_lingzhiData(lingzhi_x_Path, lingzhi_y_Path, '1')
    print(data)
