import pandas as pd
import numpy as np
import math
from scipy import sparse
from tqdm.auto import tqdm
from .generic_loader import GenericLoader

class HotspotLoader(GenericLoader):
    def __init__(self, dataset_name, data_location, features, start='2000-11-01', end='2020-10-20', post_process = None):
        GenericLoader.__init__(self)

        # self.data_location = "/home/blair/peat/data/VIIRS/fire_nrt_J1V-C2_161837.csv"
        self.spatial_res = 0.01
        self.static = False
        self.dataset_name = dataset_name
        self.data = pd.read_csv(data_location).set_index('acq_date')
        self.features = features
        self.start_date = start
        self.end_date = end
        
        if (post_process is not None):
            self.data = post_process(self.data)
        
    def load_data(self):
        # return data as numpy array with shape
        # (data x features x latitude x longitude)
        # if static, (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        # input list of features
        dates = self.get_date_range(self.start_date, self.end_date)
        data = np.stack([self.load_date_data(date)[0] for date in tqdm(dates)], axis=0)
        return (data, dates, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        # return data as numpy array with shape
        # (features x latitude x longitude)
        date_data = np.stack([self.to_points(self.data[self.data.index == date], feature) for feature in self.features], axis=0)
        return (date_data, self.features)
    
    def to_points(self, df, value, spatial_res = 0.1):
        total_x = math.ceil((self.N_LIMIT - self.S_LIMIT) / spatial_res)
        total_y = math.ceil((self.E_LIMIT - self.W_LIMIT) / spatial_res)
        boxed_df = df[(df.longitude > self.W_LIMIT) & (df.longitude < self.E_LIMIT) & (df.latitude > self.S_LIMIT) & (df.latitude < self.N_LIMIT)]
        lats = (boxed_df.latitude.to_numpy() - self.S_LIMIT) / spatial_res
        lngs = (boxed_df.longitude.to_numpy() - self.W_LIMIT) / spatial_res
        values = boxed_df[value].to_numpy()
        arr = sparse.coo_matrix((values,(lats.astype(int),lngs.astype(int))),shape=(total_x, total_y))

        dok=sparse.dok_matrix((arr.shape),dtype=arr.dtype)
        dok._update(zip(zip(arr.row,arr.col),arr.data))
        return np.array(dok.todense())
    
class J1VIIRSLoader(HotspotLoader):
    def __init__(self, start_date='2000-11-01', end_date='2020-10-20'):
        data_location = "/home/blair/peat/data/hotspots/J1VIIRS/fire_nrt_J1V-C2_161837.csv"
        features = ['bright_ti4', 'confidence', 'frp']
        HotspotLoader.__init__(self, "J1VIIRS", data_location, features, start_date, end_date)

class VIIRSLoader(HotspotLoader):
    def __init__(self, start_date = "2012-01-20", end_date = "2020-05-31"):
        data_location = "/home/blair/peat/data/hotspots/VIIRS/fire_archive_V1_161838.csv"
        features = ['bright_ti4', 'confidence', 'frp']
        def post_process(df):
            df.confidence = df.confidence.apply(lambda x: {'l': 0, 'n': 1, 'h': 2}[x])
            return df
        HotspotLoader.__init__(self, "VIIRS", data_location, features, start_date, end_date, post_process)

class MODISLoader(HotspotLoader):
    def __init__(self, start_date = "2000-11-01", end_date = "2020-06-30"):
        data_location = "/home/blair/peat/data/hotspots/MODIS/fire_archive_M6_161836.csv"
        features = ['brightness', 'confidence', 'frp']
        HotspotLoader.__init__(self, "MODIS", data_location, features, start_date, end_date)
        