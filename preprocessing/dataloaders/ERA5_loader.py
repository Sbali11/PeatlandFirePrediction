from datetime import datetime
import numpy as np
import cfgrib
import cdsapi
from . import helper
from load_data import GenericLoader

# API for Dataloader loaders
class ERA5Loader(GenericLoader):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 0.1
        self.static = False
        
        self.date_range = self.get_date_range("2000-01-01", "2020-06-30") # "2019-01-31","2019-02-01")#"1981-01-01", 
        self.dataset_name = "ERA5"
        self.features = ['t2m', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'stl1', 'stl2', 'stl3', 'stl4', 'lai_lv', 'lai_hv', 'tp', 'u10', 'v10']
        self.keys = []
        for feature in self.features:
            key = feature
            if feature == "lai_lv":
                key = "LEAF AREA INDEX LOW VEGETATION"
            elif feature == "lai_hv":
                key = "LEAF AREA INDEX HIGH VEGETATION"
            elif feature == "u10":
                key = 165
            elif feature == "v10":
                key = 166
            elif feature == "t2m":
                key = 167
            self.keys.append(key)
           
        self.cds = cdsapi.Client()
        """
        self.feature_sets = {}
        for key, feature in zip(self.keys, self.features):
            # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
            grib_file = '/DataSet/peat/ERA5/{}.grib'.format(feature)
            cds = cdsapi.Client()
            cds.retrieve('reanalysis-era5-land', {
                "product_type": "reanalysis",
                "area": "90.00/-141.00/41.75/-50.00", # NWSE
                "variable": [key],
                "year": [str(year) for year in range(2010, 2019)], # limit request size
                "month": ['{:0>2}'.format(month) for month in range(1, 12)],
                "day" : ['{:0>2}'.format(day) for day in range(1, 31)],
                "time": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
                        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", 
                        "20", "21", "22", "23"],
                "format": "grib"
            }, grib_file)

            # read grib
            ds = cfgrib.open_datasets(grib_file)
            D = ds[0]
            self.feature_sets[feature] = D
        """
        
    def load_data(self):
        # return data as numpy array with shape
        # (data x features x latitude x longitude)
        # if static, (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        # input list of features
        time_layers = []
        for d in self.date_range:
            print(d)
            time_layers.append(self.load_date_data(d))
            
        data = np.stack(time_layers, axis=0)
        return (data, self.date_range, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        # return data as numpy array with shape
        # (features x latitude x longitude)
        # get variable
        dt = datetime.strptime(date, '%Y-%m-%d')
        
        layers = []
        for key, feature in zip(self.keys, self.features):
            print(feature)
            # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
            grib_file = '/DataSet/peat/ERA5/{}_{:0>2}_{}_{}.grib'.format(dt.year, dt.month, dt.day, feature)
            self.cds.retrieve('reanalysis-era5-land', {
                "product_type": "reanalysis",
                "area": "90.00/-141.00/41.75/-50.00", # NWSE
                "variable": [key],
                "year": [str(dt.year)],
                "month": ['{:0>2}'.format(dt.month)],
                "day" : ['{:0>2}'.format(dt.day)],
                "time": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
                        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", 
                        "20", "21", "22", "23"],
                "format": "grib"
            }, grib_file)

            # read grib
            ds = cfgrib.open_datasets(grib_file)
            D = ds[0]
            
            #D = self.feature_sets[feature]

            # get array
            A = D[feature]
            if feature in ["lai_lv", "lai_hv", "tp", "u10", "v10"]:
                A = A[1, :, :, :]

            # get mean along steps
            if feature == "tp": # aggregate, https://apps.ecmwf.int/codes/grib/param-db?id=228
                A = np.nansum(A, axis=0)
            else: # average
                A = np.nanmean(A, axis=0)
            layers.append(A)
        
        data = np.stack(layers, axis=0)
        return data