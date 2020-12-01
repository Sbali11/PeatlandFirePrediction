from . import helper
from .generic_loader import GenericLoader
from netCDF4 import Dataset
from os import listdir
import numpy as np
from os.path import isfile, join
import pandas as pd
import tables
from scipy.io import loadmat

class SMAPLoader(GenericLoader):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = (3,2) #lons,lats
        self.static = False
        self.dataset_name = "MTDCA"
        self.features = 'MTDCA'
        
        path = "/home/blair/peat/data/CT2009B_XCO2/data/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if "glb" in f]
        onlyfiles = sorted(onlyfiles)
        dates = []
        data = []
        for f in onlyfiles:
            dates.append(f[24:-3])
            with Dataset(join(path,f), "r", format="NETCDF4") as f:
                data.append(np.average(np.array(f.variables['co2'][:]),axis = 0))
        with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
            self.lons = f.variables['longitude'][:]
            self.lats = f.variables['latitude'][:]
            self.unit = f.variables['co2'].units
        data = np.vstack(data)

        #check for holes
        ref_date_range = pd.date_range(dates[0], dates[-1], freq='1d')
        df = pd.DataFrame(dates)
        ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
        ref_df.index = ref_date_range
        missing_dates = ref_df.index[~ref_df.index.isin(df[0])]
        assert(len(missing_dates)==0)
        self.dates, self.data = dates, data

    def load_data(self):
        return (self.data, self.dates, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        return self.data[self.dates.index(date)]
    
    def load_smap(self, month, year, **kwargs):
        start_month = ((month - 1) // 3) * 3 + 1
        end_month = start_month + 2
        file = tables.open_file('../../data/SMAP/MTDCA_V4_SM_{}{:0>2}_{}{:0>2}_9km.mat'.format(year, start_month, year, end_month))
        date_indexes = np.where(file.root.DateVector[1] == month)[0]
        start_date, end_date = date_indexes[0], date_indexes[-1] + 1
        tmp_DCA = np.copy(file.root.MTDCA_SM[start_date: end_date])
        data = np.nanmean(tmp_DCA,axis=0).T[710:907,2938:3216]
        
        coords = loadmat('data/SMAP/SMAPCenterCoordinates9KM.mat')
        # latlng_data = interpolate.latlng_interpolate(data, coords['SMAPCenterLatitudes'][710:907,2938:3216], coords['SMAPCenterLongitudes'][710:907,2938:3216], 0.01)

        # return latlng_data
