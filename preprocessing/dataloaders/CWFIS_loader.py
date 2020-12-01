import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from . import helper
from load_data import GenericLoader

class CWFISLoader(GenericLoader):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 0.01
        self.static = False
        self.dataset_name = "CWFIS"
        
        # technically goes back from 1986-04-03, but will start at 2000 as that is probably when data is more reliable
        self.date_range = self.get_date_range("2000-01-01", "2019-12-31")
        self.burned_features = ["BURNCLAS", "FIRECAUS"]
        self.fire_features = ["SIZE_HA", "CALC_HA"]
        self.features = ["burned_pct_cat", "burned_cause", "fire_size_agency", "fire_size_calc"]
        self.cached_shape = (4824, 9099) # from Tarnocai of same "resolution"
        
        # burned
        path = "/DataSet/peat/CWFIS/NFDB_burned/nbac_1986_to_2019_20200921.shp"
        burned_shapefile = gpd.read_file(path)
        burned_shapefile = burned_shapefile.to_crs(epsg=4326) # WGS84 Latitude/Longitude
        self.burned_shapefile = burned_shapefile
        self.burned_bounds = None # cache for resolution
        
        # fire
        path = "/DataSet/peat/CWFIS/NFDB_poly/NFDB_poly_20201005.shp"
        fire_shapefile = gpd.read_file(path)
        fire_shapefile = fire_shapefile.to_crs(epsg=4326) # WGS84 Latitude/Longitude
        fire_shapefile = fire_shapefile[fire_shapefile["REP_DATE"].notnull()] # throw out points without date
        self.fire_shapefile = fire_shapefile
        self.fire_bounds = None # cache for resolution
        
        """ compute average duration of fire
        known_interval = fire_shapefile[fire_shapefile["OUT_DATE"].notnull()]
        rep_date = known_interval["REP_DATE"].tolist()
        out_date = known_interval["OUT_DATE"].tolist()
        rep_date = [datetime.datetime.strptime(r, '%Y-%m-%d') for r in rep_date]
        out_date = [datetime.datetime.strptime(o, '%Y-%m-%d') for o in out_date]
        diff = [(o-r).total_seconds() for r,o in zip(rep_date, out_date)]
        avg_secs = sum(diff) / float(len(diff))
        print(avg_secs/3600/24)
        """
        self.avg_duration = relativedelta(seconds=-1887275.9041803663) # roughly 21 days
 
    def load_data(self):
        # return data as numpy array with shape
        # (data x features x latitude x longitude)
        # if static, (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        time_layers = [self.load_date_data(d) for d in self.date_range]
        data = np.stack(time_layers, axis=0)
        return (data, self.date_range, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        # return data as numpy array with shape
        # (features x latitude x longitude)
        # West, -141.0000; East, -50.0000; North, 90.0000; South, 41.7500
        start_date = date
        s_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        e_date = s_date + relativedelta(days=1)
        end_date = e_date.strftime("%Y-%m-%d")
        layers = []
        for f in self.burned_features:
            layer = self._load_CWFIS_burned(start_date, end_date, f)
            layers.append(layer)
        for f in self.fire_features:
            layer = self._load_CWFIS_fire(start_date, end_date, f)
            layers.append(layer)
            
        # print("before", [layers.shape for layer in layers])
        layers = [layer[:self.cached_shape[0],:self.cached_shape[1]] for layer in layers]
        # print("after", [layers.shape for layer in layers])
        data = np.stack(layers, axis=0)
        return data
        
    def _load_CWFIS_burned(self, start_date, end_date, feature):
        # overlap test: https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-two-integer-ranges-for-overlap
        case1 = self.burned_shapefile[self.burned_shapefile["AFSDATE"] < end_date]
        case1 = case1[start_date <= case1["AFEDATE"]]
        case2 = self.burned_shapefile[self.burned_shapefile["SDATE"] < end_date]
        case2 = case2[start_date <= case2["EDATE"]]
        in_date = pd.concat([case1, case2], ignore_index=True)

        layer = None
        try:
            layer = helper.get_numpy(feature, in_date, self.spatial_res)
        except Exception:
            pass
        if layer is not None:
            if self.burned_bounds is None:
                affine = helper.get_affine(layer, feature)
                self.burned_bounds = helper.get_bounds(layer, affine)
            slide = layer.values
            slide = helper.bound_numpy(slide, self.burned_bounds)
        else:
            slide = np.full(self.cached_shape, np.nan)
        return slide
    
    def _load_CWFIS_fire(self, start_date, end_date, feature):
        # overlap test: https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-two-integer-ranges-for-overlap
        # can use strings to achieve same effect as time, as we order date by year/month/day
        known_interval = self.fire_shapefile[self.fire_shapefile["OUT_DATE"].notnull()]
        case1 = known_interval[known_interval["REP_DATE"] < end_date]
        case1 = case1[start_date <= case1["OUT_DATE"]]

        end_avg = datetime.datetime.strptime(end_date, "%Y-%m-%d") + self.avg_duration
        ea_date = end_avg.strftime("%Y-%m-%d")
        unknown_interval = self.fire_shapefile[self.fire_shapefile["OUT_DATE"].isnull()]
        case2 = unknown_interval[unknown_interval["REP_DATE"] < end_date]
        case2 = case2[ea_date <= case2["REP_DATE"]] # end_date <= case2["REP_DATE"] + avg_duration

        in_date = pd.concat([case1, case2], ignore_index=True)

        layer = None
        try:
            layer = helper.get_numpy(feature, in_date, self.spatial_res)
        except Exception:
            pass
        if layer is not None:
            if self.fire_bounds is None:
                affine = helper.get_affine(layer, feature)
                self.fire_bounds = helper.get_bounds(layer, affine)
            slide = layer.values
            slide = helper.bound_numpy(slide, self.fire_bounds)
        else:
            slide = np.full(self.cached_shape, np.nan)
        return slide
        