import geopandas as gpd
from . import helper
from load_data import GenericLoader
import numpy as np

class TarnocaiLoader(GenericLoader):
    def __init__(self):
        GenericLoader.__init__(self)
        # set these variables
        self.spatial_res = 0.01
        self.static = True
        self.dataset_name = "Tarnocai"
        self.features = ["peatland_pct"] # presented
        
        path = "/DataSet/peat/tarnocai/01_DATA/01_1_Shp/peat032005.shp"
        shapefile = gpd.read_file(path)
        table = "/DataSet/peat/tarnocai/01_DATA/01_1_Shp/peat032005lyr.dbf"
        table_df = gpd.read_file(table)
        table_df = table_df.drop(['geometry'], axis=1)
        shapefile = shapefile.merge(table_df, on='POLYGON_ID', how='left')

        # https://gis.stackexchange.com/questions/341827/setting-projection-parameters-in-geopandas
        shapefile = shapefile.set_crs({'proj': 'lcc', 'lat_1': 49, 'lat_2': 77, 'lat_0': 0, 'lon_0': -91.86667, 'x_0': 0, 'y_0': 0, 'datum': 'NAD27', 'units': 'm', 'no_defs': True})
        shapefile = shapefile.to_crs(epsg=4326) 
        
        self.features = ["PEATLAND_P", "TOCC", "BOG_PCT", "FEN_PCT", "SWAMP_PCT", "MARSH_PCT"]
        layers = []
        for feature in self.features:
            layer = helper.get_numpy(feature, shapefile, self.spatial_res)
            affine = helper.get_affine(layer, feature) # can store in cache or something
            bounds = helper.get_bounds(layer, affine)
            slide = layer.values
            slide = helper.bound_numpy(slide, bounds)
            layers.append(slide)
        self.data = np.stack(layers, axis=0)

    def load_data(self):
        return (self.data, None, self.features)
    
    def load_date_data(self, date='2020-10-20'):
        return self.data