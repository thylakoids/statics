class Config:
    pass


class WindowsConfig(Config):
    chromatographyDataPath = 'C:/Users/clay/Dropbox/statistics/regression/data/chromatographydata_fn_2.xlsx'


class MacConfig(Config):
    chromatographyDataPath = '/Users/dracarys/Dropbox/statistics/regression/data/chromatographydata_fn_2.xlsx'
    lingzhi_x_Path = '/Users/dracarys/Dropbox/statistics/regression/data/lingzhi_x.xlsx'
    lingzhi_y_Path = '/Users/dracarys/Dropbox/statistics/regression/data/lingzhi_y.xlsx'


config = {
    'Windows': WindowsConfig,
    'Mac': MacConfig,
}


conf = config['Mac']
