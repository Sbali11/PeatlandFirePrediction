from . import helper
from .generic_loader import GenericLoader
from netCDF4 import Dataset
from os import listdir
import numpy as np
from os.path import isfile, join
import pandas as pd

class CT2019B(GenericLoader):
    def __init__(self, path, key, date_chunk, feature, freq='1d'):
        # pass
        # path = "/DataSet/peat/CT2009B_XCO2/data/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if key in f]
        onlyfiles = sorted(onlyfiles)
        dates = []
        data = []
        self.errs = []
        assert(len(onlyfiles)>0)
        for f in onlyfiles:
            dates.append(f[date_chunk[0]:date_chunk[1]])
            try:
                with Dataset(join(path,f), "r+", format="NETCDF4") as f:
                    data.append(np.average(np.array(f.variables[feature][:]),axis = 0))
            except:
                self.errs.append(f[date_chunk[0]:date_chunk[1]])
                print("Warning: Error encountered on Date: {} at {}".format(f[date_chunk[0]:date_chunk[1]], join(path,f)))
                data.append(np.full_like(data[0], np.nan, dtype=data[0].dtype))
        with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
            self.lons = f.variables['longitude'][:]
            self.lats = f.variables['latitude'][:]
            self.unit = f.variables[feature].units
        if len(data[0].shape)<3:
            data = np.stack(data)[:,np.newaxis,:,:]
        else:
            data = np.stack(data)
        
        #check for holes
        ref_date_range = pd.date_range(dates[0], dates[-1], freq=freq)
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

class CT2019B_XCO2_global(CT2019B):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = (3,2) #lons,lats
        self.static = False
        self.dataset_name = "CT2019B_XCO2_global"
        self.features = ["height_{}".format(x) for x in range(25)] # presented
        
        path = "/DataSet/peat/CT2009B_XCO2/data/"
        super().__init__(path, 'glb', [24, -3], 'co2')
        # onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if "glb" in f]
        # onlyfiles = sorted(onlyfiles)
        # dates = []
        # data = []
        # for f in onlyfiles:
        #     dates.append(f[24:-3])
        #     with Dataset(join(path,f), "r+", format="NETCDF4") as f:
        #         data.append(np.average(np.array(f.variables['co2'][:]),axis = 0))
        # with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
        #     self.lons = f.variables['longitude'][:]
        #     self.lats = f.variables['latitude'][:]
        #     self.unit = f.variables['co2'].units
        # data = np.stack(data)

        # #check for holes
        # ref_date_range = pd.date_range(dates[0], dates[-1], freq='1d')
        # df = pd.DataFrame(dates)
        # ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
        # ref_df.index = ref_date_range
        # missing_dates = ref_df.index[~ref_df.index.isin(df[0])]
        # assert(len(missing_dates)==0)
        # self.dates, self.data = dates, data

class CT2019B_XCO2_nm(CT2019B):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 1
        self.static = False
        self.dataset_name = "CT2019B_XCO2_north_america"
        self.features = ["height_{}".format(x) for x in range(25)] # presented
        
        path = "/DataSet/peat/CT2009B_XCO2/data/"
        super().__init__(path, 'glb', [24, -3], 'co2')
        # onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if "nam" in f]
        # onlyfiles = sorted(onlyfiles)
        # dates = []
        # data = []
        # for f in onlyfiles:
        #     dates.append(f[24:-3])
        #     with Dataset(join(path,f), "r", format="NETCDF4") as f:
        #         data.append(np.average(np.array(f.variables['co2'][:]),axis = 0))
        # with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
        #     self.lons = f.variables['longitude'][:]
        #     self.lats = f.variables['latitude'][:]
        #     self.unit = f.variables['co2'].units
        # data = np.stack(data)

        # #check for holes
        # ref_date_range = pd.date_range(dates[0], dates[-1], freq='1d')
        # df = pd.DataFrame(dates)
        # ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
        # ref_df.index = ref_date_range
        # missing_dates = ref_df.index[~ref_df.index.isin(df[0])]
        # assert(len(missing_dates)==0)
        # self.dates, self.data = dates, data

class CT2019B_flux_fire(CT2019B):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 1
        self.static = False
        self.dataset_name = "CT2019B_flux"
        self.features = ["flux_fire"] # presented
        
        path = "/DataSet/peat/CT2009B_flux/data/"
        super().__init__(path, 'flux1x1', [16, -3], 'fire_flux_imp')
        # onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if "flux1x1" in f]
        # onlyfiles = sorted(onlyfiles)
        # dates = []
        # data = []
        # for f in onlyfiles:
        #     dates.append(f[16:-3])
        #     with Dataset(join(path,f), "r", format="NETCDF4") as f:
        #         data.append(np.average(np.array(f.variables['fire_flux_imp'][:]),axis = 0))
        # with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
        #     self.lons = f.variables['longitude'][:]
        #     self.lats = f.variables['latitude'][:]
        #     self.unit = f.variables['fire_flux_imp'].units
        # data = np.stack(data)[:,np.newaxis,:,:]

        # #check for holes
        # ref_date_range = pd.date_range(dates[0], dates[-1], freq='1d')
        # df = pd.DataFrame(dates)
        # ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
        # ref_df.index = ref_date_range
        # missing_dates = ref_df.index[~ref_df.index.isin(df[0])]
        # assert(len(missing_dates)==0)
        # self.dates, self.data = dates, data

class CT2019B_flux_fuel(CT2019B):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 1
        self.static = False
        self.dataset_name = "CT2019B_flux"
        self.features = ["flux_fuel"] # presented
        
        path = "/DataSet/peat/CT2009B_flux/data/"
        super().__init__(path, 'flux1x1', [16, -3], 'fossil_flux_imp')
        # onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if "flux1x1" in f]
        # onlyfiles = sorted(onlyfiles)
        # dates = []
        # data = []
        # for f in onlyfiles:
        #     dates.append(f[16:-3])
        #     with Dataset(join(path,f), "r", format="NETCDF4") as f:
        #         data.append(np.average(np.array(f.variables['fossil_flux_imp'][:]),axis = 0))
        # with Dataset(join(path,onlyfiles[0]), "r", format="NETCDF4") as f:
        #     self.lons = f.variables['longitude'][:]
        #     self.lats = f.variables['latitude'][:]
        #     self.unit = f.variables['fossil_flux_imp'].units
        # data = np.stack(data)[:,np.newaxis,:,:]

        # #check for holes
        # ref_date_range = pd.date_range(dates[0], dates[-1], freq='1d')
        # df = pd.DataFrame(dates)
        # ref_df = pd.DataFrame(np.random.randint(1, 20, (ref_date_range.shape[0], 1)))
        # ref_df.index = ref_date_range
        # missing_dates = ref_df.index[~ref_df.index.isin(df[0])]
        # assert(len(missing_dates)==0)
        # self.dates, self.data = dates, data