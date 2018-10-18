class Config:
    pass
class WindowsConfig(Config):
    chromatographyDataPath = 'C:/Users/clay/Dropbox/statics/regression/data/chromatographydata_fn_2.xlsx'
class MacConfig(Config):
    chromatographyDataPath = '/Users/dracarys/Dropbox/statics/regression/data/chromatographydata_fn_2.xlsx'


config = {
    'Windows': WindowsConfig,
    'Mac': MacConfig,
}


conf = config['Mac']



